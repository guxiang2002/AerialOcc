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

from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import os
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class DeformSelfAttention(BaseModule):
    """An attention module used in VoxFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 query_chunk_size=None,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.query_chunk_size = (
            None if query_chunk_size is None else int(query_chunk_size)
        )
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        shared_value = value is None
        if shared_value:
            assert self.batch_first
            bs, len_bev, c = query.shape
            history_query = query
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        if not shared_value:
            history_query = value[:bs]
        value = self.value_proj(value)

        if key_padding_mask is not None:
            if shared_value and key_padding_mask.shape[0] != value.shape[0]:
                key_padding_mask = key_padding_mask[:bs]
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value_batch = bs if shared_value else bs * self.num_bev_queue
        value = value.reshape(value_batch, num_value, self.num_heads, -1)

        chunk_size = self.query_chunk_size or num_query
        chunk_size = min(max(chunk_size, 1), num_query)
        output_chunks = []
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

        for start in range(0, num_query, chunk_size):
            end = min(start + chunk_size, num_query)
            chunk_queries = torch.cat(
                [history_query[:, start:end], query[:, start:end]], dim=-1)
            chunk_num_query = end - start

            sampling_offsets = self.sampling_offsets(chunk_queries).view(
                bs, chunk_num_query, self.num_heads, self.num_bev_queue,
                self.num_levels, self.num_points, 2)
            attention_weights = self.attention_weights(chunk_queries).view(
                bs, chunk_num_query, self.num_heads, self.num_bev_queue,
                self.num_levels * self.num_points)
            attention_weights = attention_weights.softmax(-1).view(
                bs, chunk_num_query, self.num_heads, self.num_bev_queue,
                self.num_levels, self.num_points)

            attention_weights = attention_weights.permute(
                0, 3, 1, 2, 4, 5).contiguous()
            sampling_offsets = sampling_offsets.permute(
                0, 3, 1, 2, 4, 5, 6).contiguous()

            chunk_reference_points = reference_points[:, start:end].reshape(
                bs, self.num_bev_queue, chunk_num_query,
                self.num_levels, -1)
            if chunk_reference_points.shape[-1] == 2:
                sampling_locations = (
                    chunk_reference_points[:, :, :, None, :, None, :]
                    + sampling_offsets
                    / offset_normalizer[None, None, None, None, :, None, :]
                )
            elif chunk_reference_points.shape[-1] == 4:
                sampling_locations = (
                    chunk_reference_points[:, :, :, None, :, None, :2]
                    + sampling_offsets / self.num_points
                    * chunk_reference_points[:, :, :, None, :, None, 2:]
                    * 0.5
                )
            else:
                raise ValueError(
                    f'Last dim of reference_points must be 2 or 4, but get '
                    f'{chunk_reference_points.shape[-1]} instead.')

            queue_outputs = []
            for queue_index in range(self.num_bev_queue):
                queue_value = value if shared_value else value[
                    queue_index::self.num_bev_queue]
                queue_locations = sampling_locations[:, queue_index]
                queue_weights = attention_weights[:, queue_index]
                if torch.cuda.is_available() and value.is_cuda:
                    def run_cuda_attention(
                            attention_value, attention_locations,
                            attention_weights):
                        return MultiScaleDeformableAttnFunction_fp32.apply(
                            attention_value, spatial_shapes, level_start_index,
                            attention_locations, attention_weights,
                            self.im2col_step)

                    checkpoint_attention = os.environ.get(
                        'CGFORMER_CHECKPOINT_ATTENTION', '1') == '1'
                    if (self.training and torch.is_grad_enabled()
                            and checkpoint_attention):
                        queue_output = checkpoint(
                            run_cuda_attention, queue_value,
                            queue_locations, queue_weights)
                    else:
                        queue_output = run_cuda_attention(
                            queue_value, queue_locations, queue_weights)
                else:
                    queue_output = multi_scale_deformable_attn_pytorch(
                        queue_value, spatial_shapes,
                        queue_locations, queue_weights)
                queue_outputs.append(queue_output)

            chunk_output = torch.stack(queue_outputs, dim=0).mean(dim=0)
            output_chunks.append(chunk_output)

        output = torch.cat(output_chunks, dim=1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
