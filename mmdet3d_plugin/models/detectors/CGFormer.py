import torch
import torch.nn as nn
import os
from contextlib import contextmanager
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from .pose_temporal_alignment import PoseTemporalAligner, predict_alignment_depth
from .temporal_fusion import (
    TemporalFeatureFusion,
    resolve_center_index,
    slice_center_cam_inputs,
    slice_center_img_metas,
)


@contextmanager
def offload_large_saved_activations(min_numel=262144):
    """Save large non-parameter activations on pinned CPU memory for backward."""
    if not torch.cuda.is_available():
        yield
        return

    def pack(tensor):
        should_offload = (
            tensor.is_cuda
            and not tensor.is_leaf
            and not tensor.is_sparse
            and tensor.numel() >= min_numel
        )
        if not should_offload:
            return None, tensor
        packed = torch.empty(
            tensor.size(), dtype=tensor.dtype, layout=tensor.layout,
            pin_memory=True)
        packed.copy_(tensor)
        return tensor.device, packed

    def unpack(packed):
        device, tensor = packed
        if device is None:
            return tensor
        return tensor.to(device, non_blocking=True)

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield

@DETECTORS.register_module()
class CGFormer(BaseModule):
    def _get_name(self):
        """Display name only; preserve legacy registry and checkpoint keys."""
        return "AerialOcc"

    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        pts_bbox_head=None,
        temporal_fusion=None,
        temporal_alignment=None,
        save_occ_prob_class_id=None,
        save_occ_prob_class_ids=None,
        teacher_depth_loss_weight=0.0,
        depth_loss=False,
        train_cfg=None,
        test_cfg=None
    ):
        super().__init__()

        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)

        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        self.save_occ_prob_class_id = (
            None if save_occ_prob_class_id is None else int(save_occ_prob_class_id)
        )
        self.save_occ_prob_class_ids = (
            None
            if save_occ_prob_class_ids is None
            else tuple(int(class_id) for class_id in save_occ_prob_class_ids)
        )
        if self.save_occ_prob_class_ids is not None:
            if not self.save_occ_prob_class_ids:
                raise ValueError('save_occ_prob_class_ids must not be empty')
            if self.save_occ_prob_class_id is not None:
                raise ValueError(
                    'Configure only one of save_occ_prob_class_id and '
                    'save_occ_prob_class_ids'
                )
        self.teacher_depth_loss_weight = teacher_depth_loss_weight

        self.temporal_aligner = (
            PoseTemporalAligner(**temporal_alignment) if temporal_alignment is not None else None
        )
        if self.temporal_aligner is not None and temporal_fusion is None:
            raise ValueError('temporal_alignment requires temporal_fusion')

        if temporal_fusion is not None:
            self.temporal_fusion = TemporalFeatureFusion(
                in_channels=temporal_fusion['in_channels'],
                hidden_channels=temporal_fusion.get('hidden_channels'),
            )
        else:
            self.temporal_fusion = None

        self.depth_loss = depth_loss

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def extract_img_feat(self, img_inputs, img_metas):
        img_enc_feats = self.image_encoder(img_inputs[0])
        b, n, _, _, _ = img_enc_feats.shape
        cam_inputs = list(img_inputs[1:7])
        proc_img_metas = img_metas

        if self.temporal_fusion is not None and n > 1:
            center_index = resolve_center_index(img_metas.get('temporal_center_index', n // 2), n)
            valid_mask = None
            if self.temporal_aligner is not None:
                alignment_depth = predict_alignment_depth(self, img_enc_feats, cam_inputs, img_metas)
                img_enc_feats, valid_mask = self.temporal_aligner(
                    img_enc_feats, alignment_depth, cam_inputs,
                    img_inputs[0].shape[-2:], center_index)
            img_enc_feats = self.temporal_fusion(img_enc_feats, center_index, valid_mask=valid_mask)
            cam_inputs = slice_center_cam_inputs(cam_inputs, n, center_index)
            proc_img_metas = slice_center_img_metas(img_metas, n, center_index)

        mlp_input = self.depth_net.get_mlp_input(*cam_inputs)
        context, depth = self.depth_net([img_enc_feats] + cam_inputs + [mlp_input], proc_img_metas)
        
        if hasattr(self, 'img_view_transformer'):
            coarse_queries = self.img_view_transformer(context, depth, cam_inputs)
        else:
            coarse_queries = None

        proposal = self.proposal_layer(cam_inputs, proc_img_metas)

        # Depth nets historically returned [B*N,D,H,W] for a single view.
        # Multi-view depth keeps the explicit camera dimension [B,N,D,H,W].
        voxformer_depth = depth if depth.dim() == 5 else depth.unsqueeze(1)

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=cam_inputs,
            lss_volume=coarse_queries,
            img_metas=proc_img_metas,
            mlvl_dpt_dists=[voxformer_depth]
        )

        return x, depth, proc_img_metas
    
    def occ_encoder(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        if os.environ.get('CGFORMER_OFFLOAD_ACTIVATIONS', '1') == '1':
            with offload_large_saved_activations():
                img_voxel_feats, depth, proc_img_metas = self.extract_img_feat(
                    img_inputs, img_metas)
        else:
            img_voxel_feats, depth, proc_img_metas = self.extract_img_feat(
                img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        # A tensor's leading axis is the batch, not feature levels.
        # Preserve the complete [B,C,X,Y,Z] tensor for B > 1.
        if isinstance(voxel_feats_enc, (list, tuple)):
            voxel_feats_enc = list(voxel_feats_enc[:1])
        else:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=proc_img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        losses = dict()

        if self.depth_loss and depth is not None:
            losses['loss_depth'] = self.depth_net.get_depth_loss(proc_img_metas['gt_depths'], depth)
        if (
            self.teacher_depth_loss_weight > 0
            and depth is not None
            and 'vggt_teacher_depth' in proc_img_metas
        ):
            losses['loss_teacher_depth'] = (
                self.teacher_depth_loss_weight
                * self.depth_net.get_depth_loss(proc_img_metas['vggt_teacher_depth'], depth)
            )

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output['output_voxels'],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output['output_voxels']
        pred = torch.argmax(pred, dim=1)

        train_output = {
            'losses': losses,
            'pred': pred,
            'gt_occ': gt_occ
        }

        return train_output
    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        gt_occ = data_dict['gt_occ']

        img_voxel_feats, depth, proc_img_metas = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        # A tensor's leading axis is the batch, not feature levels.
        # Preserve the complete [B,C,X,Y,Z] tensor for B > 1.
        if isinstance(voxel_feats_enc, (list, tuple)):
            voxel_feats_enc = list(voxel_feats_enc[:1])
        else:
            voxel_feats_enc = [voxel_feats_enc]
        
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=proc_img_metas,
            img_feats=None,
            gt_occ=gt_occ
        )

        occ_logits = output['output_voxels']
        pred = torch.argmax(occ_logits, dim=1)

        test_output = {
            'pred': pred,
            'gt_occ': gt_occ
        }
        if self.save_occ_prob_class_ids is not None:
            class_ids = self.save_occ_prob_class_ids
            invalid_ids = [
                class_id
                for class_id in class_ids
                if class_id < 0 or class_id >= occ_logits.shape[1]
            ]
            if invalid_ids:
                raise ValueError(
                    f'save_occ_prob_class_ids contains out-of-range values '
                    f'{invalid_ids} for occ logits with {occ_logits.shape[1]} classes'
                )
            probabilities = torch.softmax(occ_logits, dim=1)
            test_output['pred_occ_prob'] = probabilities[:, class_ids].sum(dim=1)
            test_output['pred_occ_prob_class_ids'] = class_ids
        elif self.save_occ_prob_class_id is not None:
            class_id = self.save_occ_prob_class_id
            if class_id < 0 or class_id >= occ_logits.shape[1]:
                raise ValueError(
                    f'save_occ_prob_class_id={class_id} out of range for '
                    f'occ logits with {occ_logits.shape[1]} classes'
                )
            test_output['pred_occ_prob'] = torch.softmax(occ_logits, dim=1)[:, class_id]
            test_output['pred_occ_prob_class_id'] = class_id

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
