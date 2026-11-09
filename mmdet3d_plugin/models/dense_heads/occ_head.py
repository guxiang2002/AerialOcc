import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from mmdet.models import HEADS
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmdet3d_plugin.utils.occupancy_losses import geo_scal_loss, sem_scal_loss, CE_ssc_loss, focal_ssc_loss

@HEADS.register_module()
class OccHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channel,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=[256, 256, 32],
        loss_weight_cfg=None,
        balance_cls_weight=True,
        class_weight_multipliers=None,
        focal_gamma=2.0,
        binary_class_id=None,
        binary_loss_weight=0.0,
        binary_pos_weight=1.0,
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=None,
        train_cfg=None,
        test_cfg=None
    ):
        super(OccHead, self).__init__()
        
        if type(in_channels) is not list:
            in_channels = [in_channels]
        
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_level = num_level
        self.empty_idx = empty_idx

        self.with_cp = with_cp
        
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        self.occ_size = occ_size
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_focal_weight = self.loss_weight_cfg.get('loss_voxel_focal_weight', 0.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.focal_gamma = focal_gamma
        self.binary_class_id = None if binary_class_id is None else int(binary_class_id)
        self.binary_loss_weight = float(binary_loss_weight)
        self.binary_pos_weight = float(binary_pos_weight)

        self.occ_convs = nn.ModuleList()
        for i in range(self.num_level):
            mid_channel = self.in_channels[i] // 2
            occ_conv = nn.Sequential(
                build_conv_layer(conv_cfg, in_channels=self.in_channels[i],
                    out_channels=mid_channel, kernel_size=3, stride=1, padding=1),
                build_norm_layer(norm_cfg, mid_channel)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(conv_cfg, in_channels=mid_channel, 
                    out_channels=out_channel, kernel_size=1, stride=1, padding=0),
            )
            self.occ_convs.append(occ_conv)
        
        # loss functions
        if balance_cls_weight:
            if class_frequencies is None:
                raise ValueError('class_frequencies must be provided when balance_cls_weight=True')
            class_weights = 1 / np.log(np.array(class_frequencies, dtype=np.float64) + 1.001)
            if class_weight_multipliers is not None:
                class_weight_multipliers = np.asarray(class_weight_multipliers, dtype=np.float64)
                if class_weight_multipliers.shape[0] != out_channel:
                    raise ValueError('class_weight_multipliers must match out_channel')
                class_weights = class_weights * class_weight_multipliers
            class_weights = class_weights / np.mean(class_weights)
            self.class_weights = torch.from_numpy(class_weights.astype(np.float32))
        else:
            self.class_weights = torch.ones(out_channel, dtype=torch.float32)
    
    def forward(self, voxel_feats, img_metas=None, img_feats=None, gt_occ=None):
        assert type(voxel_feats) is list and len(voxel_feats) == self.num_level

        output_occs = []
        for feats, occ_conv in zip(voxel_feats, self.occ_convs):
            if self.with_cp:
                output_occs.append(torch.utils.checkpoint.checkpoint(occ_conv, feats))
            else:
                output_occs.append(occ_conv(feats))
        
        result = {
            'output_voxels': F.interpolate(output_occs[0], size=self.occ_size, mode='trilinear', align_corners=False).contiguous()
        }
        return result
    
    def loss(self, output_voxels, target_voxels):
        loss_dict = {}
        loss_dict['loss_voxel_ce'] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
        if self.loss_voxel_focal_weight > 0:
            loss_dict['loss_voxel_focal'] = self.loss_voxel_focal_weight * focal_ssc_loss(
                output_voxels,
                target_voxels,
                self.class_weights.type_as(output_voxels),
                ignore_index=255,
                gamma=self.focal_gamma,
            )
        with autocast(enabled=False):
            # Keep one shared half-precision probability volume, then let the
            # scaling losses cast only each active class slice before global
            # reductions. This preserves FP32 BCE/reduction numerics without
            # materializing a full FP32 logits copy at high voxel resolution.
            output_probs = F.softmax(output_voxels, dim=1)
            loss_dict['loss_voxel_sem_scal'] = self.loss_voxel_sem_scal_weight * sem_scal_loss(
                output_probs, target_voxels, ignore_index=255, pred_is_prob=True)
            loss_dict['loss_voxel_geo_scal'] = self.loss_voxel_geo_scal_weight * geo_scal_loss(
                output_probs,
                target_voxels,
                ignore_index=255,
                non_empty_idx=self.empty_idx,
                pred_is_prob=True,
            )
        if self.binary_loss_weight > 0:
            if self.binary_class_id is None:
                raise ValueError('binary_class_id is required when binary_loss_weight > 0')
            valid = target_voxels != 255
            positive = (target_voxels == self.binary_class_id).type_as(output_voxels)
            class_logit = output_voxels[:, self.binary_class_id]
            other_logits = torch.cat(
                [
                    output_voxels[:, :self.binary_class_id],
                    output_voxels[:, self.binary_class_id + 1:],
                ],
                dim=1,
            )
            binary_logit = class_logit - torch.logsumexp(other_logits, dim=1)
            pos_weight = output_voxels.new_tensor(self.binary_pos_weight)
            loss_dict['loss_voxel_binary'] = self.binary_loss_weight * F.binary_cross_entropy_with_logits(
                binary_logit[valid], positive[valid], pos_weight=pos_weight
            )

        return loss_dict
