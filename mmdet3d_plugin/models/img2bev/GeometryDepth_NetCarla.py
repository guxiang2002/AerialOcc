import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import NECKS
from torch.cuda.amp.autocast_mode import autocast

from mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from .modules.Mono_DepthNet_modules import DepthNet
from .modules.Stereo_Depth_Net_modules import (
    Attention,
    ChannelAttention3D,
    SimpleUnet,
    SimpleUnet3D,
    convbn_2d,
)


class StereoVolumeEncoderCarla(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StereoVolumeEncoderCarla, self).__init__()
        self.stem = convbn_2d(
            in_channels, out_channels, kernel_size=3, stride=1, pad=1
        )
        self.unet = nn.Sequential(SimpleUnet(out_channels))
        self.conv_out = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.unet(x)
        x = self.conv_out(x)
        return x


class DepthAggregationCarla(nn.Module):
    def __init__(self, depth_channels, embed_dims=32, out_channels=1):
        super(DepthAggregationCarla, self).__init__()
        self.stem = nn.Conv3d(2, embed_dims, kernel_size=3, stride=1, padding=1)
        self.mono_stereo_attention = Attention(depth_channels, kernel_size=5)
        self.stereo_mono_attention = Attention(depth_channels, kernel_size=5)
        self.unet_3d = SimpleUnet3D(in_channels=embed_dims)
        self.channel_attention = ChannelAttention3D(embed_dims=embed_dims)
        self.out_conv = nn.Conv3d(
            embed_dims, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, depth_stereo, depth_mono):
        mono_stereo = self.mono_stereo_attention(depth_mono, depth_stereo)
        stereo_mono = self.stereo_mono_attention(depth_stereo, depth_mono)
        depth_feat = torch.cat(
            [mono_stereo.unsqueeze(1), stereo_mono.unsqueeze(1)], dim=1
        )
        depth_feat = F.relu(self.stem(depth_feat))
        depth_feat = self.unet_3d(depth_feat)
        depth_feat = self.channel_attention(depth_feat)
        return self.out_conv(depth_feat).squeeze(1)


@NECKS.register_module()
class GeometryDepth_NetCarla(BaseModule):
    def __init__(
        self,
        downsample=8,
        numC_input=512,
        numC_Trans=64,
        cam_channels=27,
        grid_config=None,
        loss_depth_weight=1.0,
        loss_depth_type='bce',
        use_stereo_depth=True,
    ):
        super(GeometryDepth_NetCarla, self).__init__()
        self.downsample = downsample
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.cam_channels = cam_channels
        self.grid_config = grid_config

        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.cam_depth_range = self.grid_config['dbound']
        self.stereo_volume_encoder = StereoVolumeEncoderCarla(
            in_channels=D, out_channels=D
        )
        self.depth_net = DepthNet(
            self.numC_input,
            self.numC_input,
            self.numC_Trans,
            self.D,
            cam_channels=self.cam_channels,
        )
        self.loss_depth_weight = loss_depth_weight
        self.loss_depth_type = loss_depth_type
        self.use_stereo_depth = use_stereo_depth
        self.constant_std = 0.5
        self.depth_aggregation = DepthAggregationCarla(
            depth_channels=self.D, embed_dims=32, out_channels=1
        )

    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        _, depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds, depth_labels, reduction='none'
            ).sum() / max(1.0, fg_mask.sum())
        return depth_loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(
            depth_labels,
            self.downsample,
            self.cam_depth_range,
            constant_std=self.constant_std,
        )
        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (
            depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2])
        )
        if fg_mask.sum() == 0:
            return depth_preds.new_zeros(())
        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]
        )
        if depth_gaussian_labels.numel() == 0 or depth_preds.numel() == 0:
            return depth_preds.new_zeros(())
        return F.kl_div(
            torch.log(depth_preds + 1e-4),
            depth_gaussian_labels,
            reduction='batchmean',
            log_target=False,
        )

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if depth_preds.dim() == 5:
            depth_preds = depth_preds.flatten(0, 1)
        if self.loss_depth_type == 'bce':
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)
        elif self.loss_depth_type == 'kld':
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)
        else:
            raise NotImplementedError
        return self.loss_depth_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths):
        B, N, H, W = gt_depths.shape
        valid_h = (H // self.downsample) * self.downsample
        valid_w = (W // self.downsample) * self.downsample
        if valid_h != H or valid_w != W:
            gt_depths = gt_depths[:, :, :valid_h, :valid_w].contiguous()
            H, W = valid_h, valid_w
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample,
            self.downsample,
            W // self.downsample,
            self.downsample,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample, W // self.downsample)
        gt_depths = (
            gt_depths
            - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)
        ) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(
            -1, self.D + 1
        )[:, 1:]
        return gt_depths_vals, gt_depths.float()

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        if intrin.shape[-1] == 4:
            mlp_input = torch.stack(
                [
                    intrin[:, :, 0, 0],
                    intrin[:, :, 1, 1],
                    intrin[:, :, 0, 2],
                    intrin[:, :, 1, 2],
                    intrin[:, :, 0, 3],
                    intrin[:, :, 1, 3],
                    intrin[:, :, 2, 3],
                    post_rot[:, :, 0, 0],
                    post_rot[:, :, 0, 1],
                    post_tran[:, :, 0],
                    post_rot[:, :, 1, 0],
                    post_rot[:, :, 1, 1],
                    post_tran[:, :, 1],
                    bda[:, :, 0, 0],
                    bda[:, :, 0, 1],
                    bda[:, :, 1, 0],
                    bda[:, :, 1, 1],
                    bda[:, :, 2, 2],
                ],
                dim=-1,
            )
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
            mlp_input = torch.stack(
                [
                    intrin[:, :, 0, 0],
                    intrin[:, :, 1, 1],
                    intrin[:, :, 0, 2],
                    intrin[:, :, 1, 2],
                    post_rot[:, :, 0, 0],
                    post_rot[:, :, 0, 1],
                    post_tran[:, :, 0],
                    post_rot[:, :, 1, 0],
                    post_rot[:, :, 1, 1],
                    post_tran[:, :, 1],
                    bda[:, :, 0, 0],
                    bda[:, :, 0, 1],
                    bda[:, :, 1, 0],
                    bda[:, :, 1, 1],
                    bda[:, :, 2, 2],
                ],
                dim=-1,
            )
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(
            B, N, -1
        )
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def forward(self, input, img_metas):
        x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        mono_digit = x[:, :self.D, ...]
        mono_volume = self.get_depth_dist(mono_digit)
        img_feat = x[:, self.D : self.D + self.numC_Trans, ...]

        if not self.use_stereo_depth or 'stereo_depth' not in img_metas:
            return img_feat.view(B, N, -1, H, W), mono_volume.view(B, N, self.D, H, W)

        stereo_depth = img_metas['stereo_depth']
        _, stereo_volume = self.get_downsampled_gt_depth(stereo_depth)
        stereo_volume = stereo_volume.view(B * N, H, W, -1).permute(0, 3, 1, 2)
        stereo_volume = self.stereo_volume_encoder(stereo_volume)
        stereo_volume = self.get_depth_dist(stereo_volume)

        depth_volume = self.depth_aggregation(stereo_volume, mono_volume)
        depth_volume = self.get_depth_dist(depth_volume)
        depth_volume = depth_volume.view(B, N, self.D, H, W)
        return img_feat.view(B, N, -1, H, W), depth_volume
