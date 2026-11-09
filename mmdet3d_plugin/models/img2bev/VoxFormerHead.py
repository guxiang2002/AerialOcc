# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding

@HEADS.register_module()
class VoxFormerHead(nn.Module):
    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        point_cloud_range,
        embed_dims,
        cross_transformer,
        self_transformer,
        positional_encoding,
        mlp_prior=False,
        **kwargs
    ):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims
        
        self.data_config = data_config
        self.point_cloud_range = point_cloud_range
        self.volume_embed = nn.Embedding((self.volume_h) * (self.volume_w) * (self.volume_z), self.embed_dims)
        # self.voxelize = Voxelization(point_cloud_range=point_cloud_range, spatial_shape=np.array([volume_h, volume_w, volume_z]))
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)

        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)
        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)

        if mlp_prior:
            self.mlp_prior = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dims//2, self.embed_dims)
            )
        else:
            self.mlp_prior = None
            self.mask_embed = nn.Embedding(1, self.embed_dims)

    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(
            torch.arange(self.volume_h), torch.arange(self.volume_w),torch.arange(self.volume_z), 
            indexing='ij')
        
        idx = torch.arange(self.volume_h * self.volume_w * self.volume_z)
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_3d = torch.cat(
            [(xv.reshape(-1, 1) + 0.5) / self.volume_h, 
             (yv.reshape(-1, 1) + 0.5) / self.volume_w, 
             (zv.reshape(-1, 1) + 0.5) / self.volume_z], dim=-1
        )

        return vox_coords, ref_3d

    def create_grid(self):
        # make grid in image plane
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)

    def get_positional_shape(self, num_queries):
        row_limit = getattr(getattr(self.positional_encoding, 'row_embed', None), 'num_embeddings', 512)
        col_limit = getattr(getattr(self.positional_encoding, 'col_embed', None), 'num_embeddings', 512)

        for col in range(int(col_limit), 0, -1):
            if num_queries % col != 0:
                continue
            row = num_queries // col
            if row <= row_limit:
                return int(row), int(col)

        for row in range(min(int(row_limit), num_queries), 0, -1):
            if num_queries % row != 0:
                continue
            col = num_queries // row
            if col <= col_limit:
                return int(row), int(col)

        raise ValueError(
            f'Cannot factor {num_queries} queries into positional encoding limits '
            f'row<={row_limit}, col<={col_limit}'
        )

    def _slice_batch_value(self, value, batch_idx):
        """Slice batch-aligned structures down to a single sample."""
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.dim() > 0 and value.shape[0] > batch_idx:
                return value[batch_idx:batch_idx + 1]
            return value
        if isinstance(value, (list, tuple)):
            sliced = [self._slice_batch_value(v, batch_idx) for v in value]
            return type(value)(sliced)
        if isinstance(value, dict):
            return {k: self._slice_batch_value(v, batch_idx) for k, v in value.items()}
        return value

    def _slice_meta_value(self, value, batch_idx, batch_size):
        if isinstance(value, list):
            if len(value) == batch_size:
                return value[batch_idx]
            return [self._slice_meta_value(v, batch_idx, batch_size) for v in value]
        if isinstance(value, tuple):
            if len(value) == batch_size:
                return value[batch_idx]
            return tuple(self._slice_meta_value(v, batch_idx, batch_size) for v in value)
        if torch.is_tensor(value):
            if value.dim() > 0 and value.shape[0] == batch_size:
                return value[batch_idx]
            return value
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value[batch_idx]
            return value
        return value

    def _slice_img_metas(self, img_metas, batch_idx, batch_size):
        if img_metas is None:
            return None
        if isinstance(img_metas, list):
            return [img_metas[batch_idx]]
        if isinstance(img_metas, dict):
            sample_meta = {}
            for key, value in img_metas.items():
                sample_meta[key] = self._slice_meta_value(value, batch_idx, batch_size)
            return [sample_meta]
        return [img_metas]

    def _forward_single(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None, **kwargs):
        """Original VoxFormerHead path, kept as a single-sample implementation."""
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        assert bs == 1
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device

        volume_queries = self.volume_embed.weight.to(dtype)
        lss_volume_flatten = None
        if lss_volume is not None:
            lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0)
            if lss_volume_flatten.shape[0] == volume_queries.shape[0]:
                volume_queries = volume_queries + lss_volume_flatten
            else:
                if not getattr(self, '_warned_lss_query_mismatch', False):
                    print(
                        'skip lss_volume prior due to query mismatch: '
                        f'lss={tuple(lss_volume_flatten.shape)}, '
                        f'volume_queries={tuple(volume_queries.shape)}'
                    )
                    self._warned_lss_query_mismatch = True
                lss_volume_flatten = None

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)
        # Generate positional embeddings with one position per voxel query.
        pos_h, pos_w = self.get_positional_shape(self.volume_h * self.volume_w * self.volume_z)
        bev_pos = self.positional_encoding(
            torch.zeros((bs, pos_h, pos_w), device=volume_queries.device).to(dtype)
        ).to(dtype)

        # Cross and self attention consume the same deterministic positional
        # encoding. Reusing it avoids retaining a second full query-sized
        # tensor, while these registered geometry buffers are read-only.
        bev_pos_cross_attn = bev_pos
        bev_pos_self_attn = bev_pos
        vox_coords, ref_3d = self.vox_coords, self.ref_3d
        unmasked_idx = torch.nonzero(proposal.reshape(-1) > 0).view(-1)
        masked_idx = torch.nonzero(proposal.reshape(-1) == 0).view(-1)
        # Compute seed features of query proposals by deformable cross attention
        def get_seed_features():
            return self.cross_transformer.get_vox_features(
                mlvl_feats,
                volume_queries,
                self.volume_h,
                self.volume_w,
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=None,
                bev_pos=bev_pos_cross_attn,
                img_metas=img_metas,
                prev_bev=None,
                cam_params=cam_params,
                **kwargs
            )

        seed_feats = get_seed_features()

        vox_feats = torch.empty(
            (self.volume_h, self.volume_w, self.volume_z, self.embed_dims),
            device=volume_queries.device,
            dtype=dtype,
        )
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx, 3], :] = seed_feats[0].to(dtype)
        if self.mlp_prior is None or lss_volume_flatten is None:
            if hasattr(self, 'mask_embed'):
                masked_feats = self.mask_embed.weight.view(1, self.embed_dims).expand(
                    masked_idx.shape[0], self.embed_dims).to(dtype)
            else:
                masked_feats = volume_queries[masked_idx, :]
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = masked_feats
        else:
            prior_feats = self.mlp_prior(lss_volume_flatten[masked_idx, :])
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = prior_feats.to(dtype)

        def diffuse_voxels(voxel_features):
            return self.self_transformer.diffuse_vox_features(
                mlvl_feats,
                voxel_features,
                pos_h,
                pos_w,
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=None,
                bev_pos=bev_pos_self_attn,
                img_metas=img_metas,
                prev_bev=None,
                cam_params=cam_params,
                **kwargs
            )

        if self.training and vox_feats_flatten.requires_grad:
            vox_feats_diff = checkpoint(diffuse_voxels, vox_feats_flatten)
        else:
            vox_feats_diff = diffuse_voxels(vox_feats_flatten)

        # Deformable attention accumulates in FP32 for stability. Return to the
        # surrounding AMP feature dtype only after both self-attention layers,
        # so the large persistent voxel feature does not stay FP32 downstream.
        vox_feats_diff = vox_feats_diff.to(dtype)
        vox_feats_diff = vox_feats_diff.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)
        vox_feats_diff = vox_feats_diff.permute(3, 0, 1, 2).unsqueeze(0)
        return vox_feats_diff
    
    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None,  **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information
            depth: Pre-estimated depth map, (B, 1, H_d, W_d)
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        if bs == 1:
            return self._forward_single(
                mlvl_feats,
                proposal,
                cam_params,
                lss_volume=lss_volume,
                img_metas=img_metas,
                **kwargs
            )

        outputs = []
        for batch_idx in range(bs):
            sample_feats = [feat[batch_idx:batch_idx + 1] for feat in mlvl_feats]
            sample_proposal = proposal[batch_idx:batch_idx + 1]
            sample_cam_params = self._slice_batch_value(cam_params, batch_idx)
            sample_lss_volume = self._slice_batch_value(lss_volume, batch_idx)
            sample_img_metas = self._slice_img_metas(img_metas, batch_idx, bs)
            sample_kwargs = {
                key: self._slice_batch_value(value, batch_idx)
                for key, value in kwargs.items()
            }
            outputs.append(
                self._forward_single(
                    sample_feats,
                    sample_proposal,
                    sample_cam_params,
                    lss_volume=sample_lss_volume,
                    img_metas=sample_img_metas,
                    **sample_kwargs
                )
            )

        return torch.cat(outputs, dim=0)
