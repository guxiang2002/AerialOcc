"""Opt-in, depth-aware camera-motion compensation for CARLA temporal features.

All camera extrinsics must map camera coordinates into the SAME reference frame.
Depth is optical-axis z in metres, never Euclidean range. Pixel/feature endpoints
follow this repository's LSS frustum convention (linspace and align_corners=True).
No labels, teacher maps, new learned parameters, or PanoOcc predictions are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseTemporalAligner(nn.Module):
    def __init__(self, min_depth=0.1, max_depth=1000.0,
                 depth_abs_tolerance=1.0, depth_rel_tolerance=0.05,
                 check_occlusion=True, record_diagnostics=False):
        super().__init__()
        if not 0 < min_depth < max_depth:
            raise ValueError('Require 0 < min_depth < max_depth')
        if depth_abs_tolerance < 0 or depth_rel_tolerance < 0:
            raise ValueError('Depth tolerances must be nonnegative')
        self.min_depth = float(min_depth)
        self.max_depth = float(max_depth)
        self.depth_abs_tolerance = float(depth_abs_tolerance)
        self.depth_rel_tolerance = float(depth_rel_tolerance)
        self.check_occlusion = bool(check_occlusion)
        self.record_diagnostics = bool(record_diagnostics)
        self.last_diagnostics = None

    def forward(self, features, depths, cam_inputs, image_size, center_index):
        """Return [B,N,C,H,W] aligned features and [B,N,1,H,W] validity.

        depths: [B,N,H,W] predicted camera-z maps at feature resolution.
        cam_inputs: camera-to-reference R,t; K; image post-R,post-t; optional BDA.
        BDA is intentionally unused: both views already share a pre-BDA frame.
        Only camera motion is compensated. Independently moving objects are not
        motion-warped, even when they pass the optional depth consistency check.
        """
        if features.ndim != 5 or depths.shape != features.shape[:2] + features.shape[-2:]:
            raise ValueError('Expected features B,N,C,H,W and depths B,N,H,W')
        b, n, _, h, w = features.shape
        if not 0 <= center_index < n:
            raise ValueError('center_index out of range')
        ih, iw = map(int, image_size)
        if min(ih, iw, h, w) < 2:
            raise ValueError('Image and feature spatial dimensions must exceed one')
        if len(cam_inputs) < 5:
            raise ValueError('Need R,t,K,post_R,post_t before center slicing')

        # Inversion and projective division need float32 even under mixed precision.
        with torch.cuda.amp.autocast(enabled=False):
            rots, trans, intrinsic, post_rots, post_trans = [
                v.to(device=features.device, dtype=torch.float32) for v in cam_inputs[:5]]
            if rots.shape != (b, n, 3, 3) or trans.shape != (b, n, 3):
                raise ValueError('Extrinsics must contain every temporal camera')
            if intrinsic.shape[:2] != (b, n) or intrinsic.shape[-2:] not in ((3, 3), (3, 4), (4, 4)):
                raise ValueError('Unsupported intrinsic shape')
            if intrinsic.shape[-1] == 4 and torch.any(intrinsic[..., :3, 3] != 0):
                raise ValueError('Projection translations are unsupported; pass camera K and extrinsics separately')
            intrinsic = intrinsic[..., :3, :3]
            if post_rots.shape != (b, n, 3, 3) or post_trans.shape != (b, n, 3):
                raise ValueError('Invalid image augmentation matrices')
            for matrix in (rots, trans, intrinsic, post_rots, post_trans):
                if not torch.isfinite(matrix).all():
                    raise ValueError('Nonfinite camera calibration')
            # This loader applies 2D affine image augmentation, never depth scaling.
            expected = post_rots.new_tensor([0., 0., 1.]).expand(b, n, 3)
            if not torch.allclose(post_rots[..., 2, :], expected) or torch.any(post_trans[..., 2] != 0):
                raise ValueError('Post transforms must preserve camera depth')

            depths = depths.detach().float()
            depth_valid = (torch.isfinite(depths) & (depths >= self.min_depth)
                           & (depths <= self.max_depth))
            clean_depth = torch.where(depth_valid, depths, torch.ones_like(depths))
            yy, xx = torch.meshgrid(torch.linspace(0, ih - 1, h, device=features.device),
                                    torch.linspace(0, iw - 1, w, device=features.device))
            pixels = torch.stack((xx, yy, torch.ones_like(xx)), 0).reshape(1, 3, -1).expand(b, -1, -1)
            target_pixels = torch.inverse(post_rots[:, center_index]).bmm(
                pixels - post_trans[:, center_index, :, None])
            target_camera = torch.inverse(intrinsic[:, center_index]).bmm(target_pixels)
            target_camera = target_camera * clean_depth[:, center_index].reshape(b, 1, -1)
            reference_points = rots[:, center_index].bmm(target_camera) + trans[:, center_index, :, None]
            inverse_rots = torch.inverse(rots)
            aligned, valid_masks, records = [], [], []
            center = features[:, center_index]

            for i in range(n):
                if i == center_index:
                    aligned.append(center)
                    valid_masks.append(torch.ones((b, 1, h, w), dtype=torch.bool, device=features.device))
                    if self.record_diagnostics:
                        records.append(dict(frame_index=i, valid_fraction=1.0, mean_displacement_pixels=0.0))
                    continue
                source_camera = inverse_rots[:, i].bmm(reference_points - trans[:, i, :, None])
                source_z = source_camera[:, 2].reshape(b, h, w)
                projection = intrinsic[:, i].bmm(source_camera)
                denominator = projection[:, 2:3]
                safe_denominator = torch.where(denominator.abs() > 1e-6, denominator, torch.ones_like(denominator))
                source_pixels = projection / safe_denominator
                source_pixels = post_rots[:, i].bmm(source_pixels) + post_trans[:, i, :, None]
                uv = source_pixels[:, :2].reshape(b, 2, h, w)
                grid = torch.stack((2 * uv[:, 0] / (iw - 1) - 1,
                                    2 * uv[:, 1] / (ih - 1) - 1), -1)
                valid = (depth_valid[:, center_index] & torch.isfinite(grid).all(-1)
                         & torch.isfinite(source_z) & (source_z >= self.min_depth)
                         & (source_z <= self.max_depth) & (grid.abs() <= 1 + 1e-6).all(-1))
                # Never pass NaN grids to grid_sample, including rejected locations.
                safe_grid = torch.where(torch.isfinite(grid), grid, torch.full_like(grid, 2.))
                safe_grid = torch.where(valid[..., None], safe_grid.clamp(-1., 1.), torch.full_like(safe_grid, 2.))
                if self.check_occlusion:
                    source_depth = F.grid_sample(clean_depth[:, i:i + 1], safe_grid,
                                                 align_corners=True, mode='bilinear')[:, 0]
                    source_valid = F.grid_sample(depth_valid[:, i:i + 1].float(), safe_grid,
                                                 align_corners=True, mode='bilinear')[:, 0] >= 1 - 1e-6
                    tolerance = self.depth_abs_tolerance + self.depth_rel_tolerance * source_z.abs()
                    valid = valid & source_valid & ((source_depth - source_z).abs() <= tolerance)
                sampled = F.grid_sample(features[:, i].float(), safe_grid,
                                        align_corners=True, mode='bilinear').to(features.dtype)
                aligned.append(torch.where(valid[:, None], sampled, center))
                valid_masks.append(valid[:, None])
                if self.record_diagnostics:
                    displacement = ((uv[:, 0] - xx) ** 2 + (uv[:, 1] - yy) ** 2).sqrt()
                    selected = displacement[valid]
                    records.append(dict(frame_index=i, valid_fraction=float(valid.float().mean()),
                                        mean_displacement_pixels=float(selected.mean()) if selected.numel() else None))
            self.last_diagnostics = records if self.record_diagnostics else None
            return torch.stack(aligned, 1), torch.stack(valid_masks, 1)


def predict_alignment_depth(model, features, cam_inputs, img_metas):
    """Bootstrap detached metric depth with the model's existing depth network.

    Temporarily use eval behavior so the extra geometry pass cannot update BN
    statistics or apply dropout. Restore every submodule's original training flag.
    The ordinary post-fusion depth pass remains differentiable and unchanged.
    Only the already-authorized sensor depth is forwarded as metadata; no GT or
    VGGT teacher maps enter this auxiliary inference pass.
    """
    view_transformer = getattr(model, 'img_view_transformer', None)
    if view_transformer is not None and getattr(view_transformer, 'geometry_mode', 'carla') != 'carla':
        raise ValueError('Pose alignment currently supports CARLA camera-z coordinates only')
    net = model.depth_net
    flags = [(module, module.training) for module in net.modules()]
    try:
        net.eval()
        with torch.no_grad():
            metadata = {key: img_metas[key] for key in ('stereo_depth',) if key in img_metas}
            mlp_input = net.get_mlp_input(*cam_inputs)
            _, probability = net([features.detach()] + list(cam_inputs) + [mlp_input], metadata)
            if probability.ndim != 5 or probability.shape[:2] != features.shape[:2]:
                raise ValueError('Depth network must return B,N,D,H,W for all frames')
            bins = torch.arange(*net.cam_depth_range, device=probability.device, dtype=torch.float32)
            if probability.shape[2] != bins.numel():
                raise ValueError('Depth bins do not match prediction channels')
            return (probability.float() * bins[None, None, :, None, None]).sum(2)
    finally:
        for module, was_training in flags:
            module.training = was_training
