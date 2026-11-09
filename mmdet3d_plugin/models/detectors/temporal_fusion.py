import torch
import torch.nn as nn
import numpy as np


class TemporalFeatureFusion(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels * 3, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats, center_index, valid_mask=None):
        center_feat = feats[:, center_index:center_index + 1]
        if valid_mask is None:
            # Preserve the original arithmetic and checkpoint behavior by default.
            mean_feat = feats.mean(dim=1, keepdim=True)
        else:
            if valid_mask.shape != feats.shape[:2] + (1,) + feats.shape[-2:]:
                raise ValueError('Temporal validity mask must be B,N,1,H,W')
            weights = valid_mask.to(feats.dtype)
            # Rejected projections contribute neither features nor mean weight.
            masked = torch.where(valid_mask.bool(), feats, torch.zeros_like(feats))
            mean_feat = masked.sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1)
            mean_feat = torch.where(weights.sum(dim=1, keepdim=True) > 0, mean_feat, center_feat)
        center_repeat = center_feat.expand_as(feats)
        mean_repeat = mean_feat.expand_as(feats)
        fusion_input = torch.cat([feats, center_repeat, mean_repeat], dim=2)

        b, n, c3, h, w = fusion_input.shape
        fusion_input = fusion_input.reshape(b * n, c3, h, w)
        fused = self.fuse(fusion_input).view(b, n, -1, h, w)
        fused_center = fused[:, center_index:center_index + 1]
        return fused_center


def slice_center_img_metas(img_metas, num_views, center_index):
    center_metas = {}
    for key, value in img_metas.items():
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == num_views:
            center_metas[key] = value[:, center_index:center_index + 1, ...]
        else:
            center_metas[key] = value
    return center_metas


def slice_center_cam_inputs(cam_inputs, num_views, center_index):
    center_inputs = []
    for value in cam_inputs:
        if torch.is_tensor(value) and value.dim() >= 2 and value.shape[1] == num_views:
            center_inputs.append(value[:, center_index:center_index + 1, ...])
        else:
            center_inputs.append(value)
    return center_inputs


def resolve_center_index(center_index, num_views):
    if isinstance(center_index, torch.Tensor):
        center_index = int(center_index.flatten()[0].item())
    elif isinstance(center_index, np.ndarray):
        center_index = int(center_index.reshape(-1)[0])
    elif isinstance(center_index, (list, tuple)):
        center_index = int(center_index[0])
    else:
        center_index = int(center_index)
    return max(0, min(center_index, num_views - 1))
