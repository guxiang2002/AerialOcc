import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CreateDepthFromLiDARAerialOcc(object):
    def __init__(self, data_root=None, load_seg=False, label_mapping=None, ignore_label=255):
        self.data_root = data_root
        self.load_seg = load_seg
        self.ignore_label = int(ignore_label)
        self.label_mapping = self.build_label_mapping(label_mapping)

    def build_label_mapping(self, label_mapping):
        if label_mapping is None:
            return None

        if isinstance(label_mapping, dict):
            max_label = max(max(label_mapping.keys()), self.ignore_label)
            lut = np.full(max_label + 1, self.ignore_label, dtype=np.uint8)
            for raw_label, compact_label in label_mapping.items():
                lut[int(raw_label)] = int(compact_label)
            return lut

        lut = np.asarray(label_mapping, dtype=np.uint8)
        if lut.ndim != 1:
            raise ValueError('label_mapping must be a 1D LUT or a dict')
        return lut

    def __call__(self, results):
        lidar_filename = results['lidar_path']
        lidar_points = np.fromfile(lidar_filename, dtype=np.float32).reshape(-1, 4)
        lidar_points = torch.from_numpy(lidar_points[:, :3]).float()

        if self.load_seg:
            label_filename = results.get('lidar_label_path')
            if label_filename is not None:
                label_points = np.fromfile(label_filename, dtype=np.uint32).reshape(-1, 1, 1)
                label_points = (label_points & 0xFFFF).astype(np.uint8)
                if self.label_mapping is not None:
                    invalid_mask = label_points >= self.label_mapping.shape[0]
                    label_points[invalid_mask] = self.ignore_label
                    valid_mask = ~invalid_mask
                    if np.any(valid_mask):
                        label_points[valid_mask] = self.label_mapping[label_points[valid_mask]]
                label_points = torch.from_numpy(label_points)
            else:
                label_points = None
        else:
            label_points = None

        imgs, rots, trans, intrins, post_rots, post_trans = results['img_inputs'][:6]
        projected_points = self.project_points(
            lidar_points, rots, trans, intrins, post_rots, post_trans
        )

        if label_points is not None:
            num_point, num_img, _ = projected_points.shape
            label_points = label_points.repeat(1, num_img, 1)
            projected_points = torch.cat([projected_points, label_points], dim=-1)

        img_h, img_w = imgs.shape[-2:]
        valid_mask = (
            (projected_points[..., 0] >= 0)
            & (projected_points[..., 1] >= 0)
            & (projected_points[..., 0] <= img_w - 1)
            & (projected_points[..., 1] <= img_h - 1)
            & (projected_points[..., 2] > 0)
        )

        gt_depths = []
        gt_semantics = []
        for img_index in range(imgs.shape[0]):
            gt_depth = torch.zeros((img_h, img_w))
            projected_points_i = projected_points[:, img_index]
            valid_mask_i = valid_mask[:, img_index]
            valid_points_i = projected_points_i[valid_mask_i]
            depth_order = torch.argsort(valid_points_i[:, 2], descending=True)
            valid_points_i = valid_points_i[depth_order]
            gt_depth[
                valid_points_i[:, 1].round().long(),
                valid_points_i[:, 0].round().long(),
            ] = valid_points_i[:, 2]
            gt_depths.append(gt_depth)
            if label_points is not None:
                gt_semantic = torch.full(
                    (img_h, img_w),
                    self.ignore_label,
                    dtype=valid_points_i.dtype,
                )
                gt_semantic[
                    valid_points_i[:, 1].round().long(),
                    valid_points_i[:, 0].round().long(),
                ] = valid_points_i[:, 3]
                gt_semantics.append(gt_semantic)

        gt_depths = torch.stack(gt_depths)
        results['gt_depths'] = gt_depths
        # Keep the cached input depth when it was
        # loaded earlier, and only fall back to LiDAR-projected depth when no
        # offline depth map is available.
        if 'stereo_depth' not in results:
            results['stereo_depth'] = gt_depths.clone()
        if label_points is not None:
            gt_semantics = torch.stack(gt_semantics)
            results['gt_semantics'] = gt_semantics
        return results

    def project_points(self, points, rots, trans, intrins, post_rots, post_trans):
        points = points.view(-1, 1, 3)
        points = points - trans.view(1, -1, 3)
        inv_rots = rots.inverse().unsqueeze(0)
        points = (inv_rots @ points.unsqueeze(-1))
        if intrins.shape[-1] == 4:
            points = torch.cat(
                (
                    points,
                    torch.ones((points.shape[0], points.shape[1], 1, 1)),
                ),
                dim=2,
            )
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
        else:
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)

        points_d = points[..., 2:3]
        points_uv = points[..., :2] / points_d
        points_uv = post_rots[:, :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
        points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)
        points_uvd = torch.cat((points_uv, points_d), dim=2)
        return points_uvd
