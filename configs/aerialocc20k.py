import os


data_root = os.environ.get('AERIALOCC_DATA_ROOT', 'data/AerialOcc-20K')
ann_file = os.path.join(data_root, 'preprocess')
stereo_depth_root = os.path.join(data_root, 'depth')
teacher_depth_root = os.path.join(data_root, 'vggt_depth')
camera_used = ['bev']
label_merge_mode = 'mode'
use_occ_count_mask = True
valid_threshold = 0.0
temporal_offsets = [-1, 0, 1]

dataset_type = 'AerialOcc20KDataset'
# Most AerialOcc-20K scenes use the evaluation-space bounds below. scene_00 is
# narrower in x, and is remapped into this common training space in the
# dataset loader.
point_cloud_range = [-25.6, -25.6, -32.0, 51.2, 25.6, 8.0]
occ_size = [256, 256, 32]

raw_class_names = [
    'empty',          # 0  Unlabeled
    'road',           # 1
    'sidewalk',       # 2
    'building',       # 3
    'wall',           # 4
    'fence',          # 5
    'pole',           # 6
    'traffic_light',  # 7
    'traffic_sign',   # 8
    'vegetation',     # 9
    'terrain',        # 10
    'sky',            # 11
    'pedestrian',     # 12
    'rider',          # 13
    'car',            # 14
    'truck',          # 15
    'bus',            # 16
    'train',          # 17
    'motorcycle',     # 18
    'bicycle',        # 19
    'static',         # 20
    'dynamic',        # 21
    'other',          # 22
    'water',          # 23
    'road_line',      # 24
    'ground',         # 25
    'bridge',         # 26
    'rail_track',     # 27
    'guard_rail',     # 28
]


# Fixed release class order and training histogram (empty + 26 foreground).
# The absent raw sky/train IDs are not added to the classifier.
active_label_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
raw_label_to_compact = {raw: compact for compact, raw in enumerate(active_label_ids)}
compact_to_raw_label = list(active_label_ids)
class_names = [raw_class_names[raw] for raw in active_label_ids]
num_class = len(class_names)
class_frequencies = [603389335, 9916665751, 5340712064, 7109398389, 692545594, 170525342, 385237264, 1280456, 14041691, 3836015837, 2370188666, 6568977, 844961, 459949393, 70124821, 34961546, 4234353, 920503, 219524824, 74303402, 125876532, 12495664, 433286232, 444922338, 254065123, 674501528, 18866480]


# Extra multipliers keep rare/dynamic foreground classes from being washed out
# by the dominant road/building/background voxels.
class_weight_boost_by_raw_label = {
    12: 4.0,  # pedestrian
    13: 3.0,  # rider
    14: 2.5,  # car
    15: 3.0,  # truck
    16: 4.0,  # bus
    18: 3.0,  # motorcycle
    19: 3.0,  # bicycle
    20: 1.5,  # static
    21: 2.5,  # dynamic
    24: 1.25, # road_line
}
class_weight_multipliers = [
    class_weight_boost_by_raw_label.get(raw_label_id, 1.0)
    for raw_label_id in active_label_ids
]
save_pred_as_raw_labels = True


bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.0,
    flip_dz_ratio=0.0,
)

data_config = {
    'input_size': (384, 704),
    'resize': (0.0, 0.0),
    'rot': (0.0, 0.0),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.0,
}

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        data_config=data_config,
        load_stereo_depth=True,
        load_teacher_depth=True,
        is_train=True,
        color_jitter=(0.2, 0.2, 0.2),
    ),
    dict(type='CreateDepthFromLiDARAerialOcc', data_root=data_root, load_seg=False),
    dict(
        type='LoadAnnotationOcc',
        bda_aug_conf=bda_aug_conf,
        apply_bda=False,
        is_train=True,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=[
            'pc_range',
            'occ_size',
            'sequence',
            'frame_id',
            'raw_img',
            'stereo_depth',
            'focal_length',
            'baseline',
            'img_shape',
            'gt_depths',
            'vggt_teacher_depth',
            'lidar_path',
            'temporal_center_index',
        ],
    ),
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        data_config=data_config,
        load_stereo_depth=True,
        load_teacher_depth=True,
        is_train=False,
        color_jitter=None,
    ),
    dict(type='CreateDepthFromLiDARAerialOcc', data_root=data_root, load_seg=False),
    dict(
        type='LoadAnnotationOcc',
        bda_aug_conf=bda_aug_conf,
        apply_bda=False,
        is_train=False,
        point_cloud_range=point_cloud_range,
    ),
    dict(
        type='CollectData',
        keys=['img_inputs', 'gt_occ'],
        meta_keys=[
            'pc_range',
            'occ_size',
            'sequence',
            'frame_id',
            'raw_img',
            'stereo_depth',
            'focal_length',
            'baseline',
            'img_shape',
            'gt_depths',
            'vggt_teacher_depth',
            'lidar_path',
            'temporal_center_index',
        ],
    ),
]

trainset_config = dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    teacher_depth_root=teacher_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=train_pipeline,
    split='train',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=False,
    label_merge_mode=label_merge_mode,
    camera_height=60.0,
    num_classes=num_class,
    prefer_preprocessed_labels=True,
    use_occ_count_mask=use_occ_count_mask,
    valid_threshold=valid_threshold,
    temporal_offsets=temporal_offsets,
    label_mapping=raw_label_to_compact,
)

testset_config = dict(
    type=dataset_type,
    stereo_depth_root=stereo_depth_root,
    teacher_depth_root=teacher_depth_root,
    data_root=data_root,
    ann_file=ann_file,
    pipeline=test_pipeline,
    split='val',
    camera_used=camera_used,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    test_mode=True,
    label_merge_mode=label_merge_mode,
    camera_height=60.0,
    num_classes=num_class,
    prefer_preprocessed_labels=True,
    use_occ_count_mask=use_occ_count_mask,
    valid_threshold=valid_threshold,
    temporal_offsets=temporal_offsets,
    label_mapping=raw_label_to_compact,
)

data = dict(
    train=trainset_config,
    val=testset_config,
    test=testset_config,
)

train_dataloader_config = dict(batch_size=1, num_workers=4)
test_dataloader_config = dict(batch_size=1, num_workers=4)

numC_Trans = 128
lss_downsample = [2, 2, 2]
voxel_out_channels = [128]
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    # GeometryDepth_Net uses a grouped deformable conv that requires
    # the depth-bin count D to be divisible by 4. torch.arange(1, 81, 1)
    # gives 80 bins, while torch.arange(1, 80, 1) only gives 79.
    'dbound': [1.0, 81.0, 1.0],
}

_num_layers_cross_ = 3
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 128
_pos_dim_ = _dim_ // 2

_num_layers_self_ = 2
_num_points_self_ = 8

model = dict(
    type='AerialOcc',
    temporal_fusion=dict(
        in_channels=640,
        hidden_channels=640,
    ),
    teacher_depth_loss_weight=float(os.environ.get('AERIALOCC_KD_WEIGHT', '0.1')),
    instance_geometry_weight=float(os.environ.get('AERIALOCC_GEOMETRY_WEIGHT', '1.0')),
    geometry_class_ids=(raw_label_to_compact[14],),
    img_backbone=dict(
        type='CustomEfficientNet',
        arch='b7',
        drop_path_rate=0.2,
        frozen_stages=0,
        norm_eval=False,
        out_indices=(2, 3, 4, 5, 6),
        with_cp=True,
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone',
            checkpoint=os.environ.get('AERIALOCC_B7_CHECKPOINT', './ckpts/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth'),
        ),
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[48, 80, 224, 640, 2560],
        upsample_strides=[0.5, 1, 2, 4, 4],
        out_channels=[128, 128, 128, 128, 128],
    ),
    depth_net=dict(
        type='AerialOccRelativeDepth',
        downsample=8,
        numC_input=640,
        numC_Trans=numC_Trans,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_depth_weight=1.0,
    ),
    img_view_transformer=dict(
        type='LSSViewTransformer',
        downsample=8,
        grid_config=grid_config,
        data_config=data_config,
    ),
    proposal_layer=dict(
        type='VoxelProposalLayerCarla',
        point_cloud_range=point_cloud_range,
        input_dimensions=[128, 128, 16],
        data_config=data_config,
        init_cfg=None,
    ),
    VoxFormer_head=dict(
        type='VoxFormerHead',
        volume_h=128,
        volume_w=128,
        volume_z=16,
        data_config=data_config,
        point_cloud_range=point_cloud_range,
        embed_dims=_dim_,
        cross_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams=_num_cams_,
            encoder=dict(
                type='VoxFormerEncoder_DFA3D',
                num_layers=_num_layers_cross_,
                pc_range=point_cloud_range,
                data_config=data_config,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformCrossAttention_DFA3D',
                            pc_range=point_cloud_range,
                            num_cams=_num_cams_,
                            deformable_attention=dict(
                                type='MSDeformableAttention3D_DFA3D',
                                embed_dims=_dim_,
                                num_points=_num_points_cross_,
                                num_levels=_num_levels_,
                            ),
                            embed_dims=_dim_,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=_dim_ * 2,
                    ffn_dropout=0.1,
                    operation_order=('cross_attn', 'norm', 'ffn', 'norm'),
                ),
            ),
        ),
        self_transformer=dict(
            type='PerceptionTransformer_DFA3D',
            rotate_prev_bev=True,
            use_shift=True,
            embed_dims=_dim_,
            num_cams=_num_cams_,
            use_level_embeds=False,
            use_cams_embeds=False,
            encoder=dict(
                type='VoxFormerEncoder',
                num_layers=_num_layers_self_,
                pc_range=point_cloud_range,
                data_config=data_config,
                num_points_in_pillar=8,
                return_intermediate=False,
                transformerlayers=dict(
                    type='VoxFormerLayer',
                    attn_cfgs=[
                        dict(
                            type='DeformSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            num_points=_num_points_self_,
                        )
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=_dim_,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    feedforward_channels=_dim_ * 2,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                ),
            ),
        ),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=512,
            col_num_embed=512,
        ),
        mlp_prior=True,
    ),
    occ_encoder_backbone=dict(
        type='Fuser',
        embed_dims=128,
        global_aggregator=dict(
            type='TPVGlobalAggregator',
            embed_dims=_dim_,
            split=[8, 8, 8],
            grid_size=[128, 128, 16],
            global_encoder_backbone=dict(
                type='Swin',
                embed_dims=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7,
                mlp_ratio=4,
                in_channels=128,
                patch_size=4,
                strides=[1, 2, 2, 2],
                frozen_stages=-1,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                patch_norm=True,
                out_indices=[1, 2, 3],
                with_cp=False,
                convert_weights=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=os.environ.get('AERIALOCC_SWIN_CHECKPOINT', './ckpts/swin_tiny_patch4_window7_224.pth'),
                ),
            ),
            global_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[192, 384, 768],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=dict(type='BN2d', requires_grad=True, track_running_stats=False),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='bilinear', align_corners=False),
            ),
        ),
        local_aggregator=dict(
            type='LocalAggregator',
            local_encoder_backbone=dict(
                type='CustomResNet3D',
                numC_input=128,
                num_layer=[2, 2, 2],
                num_channels=[128, 128, 128],
                stride=[1, 2, 2],
            ),
            local_encoder_neck=dict(
                type='GeneralizedLSSFPN',
                in_channels=[128, 128, 128],
                out_channels=_dim_,
                start_level=0,
                num_outs=3,
                norm_cfg=norm_cfg,
                conv_cfg=dict(type='Conv3d'),
                act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='trilinear', align_corners=False),
            ),
        ),
    ),
    pts_bbox_head=dict(
        type='OccHead',
        in_channels=[sum(voxel_out_channels)],
        out_channel=num_class,
        empty_idx=0,
        num_level=1,
        with_cp=True,
        occ_size=occ_size,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_focal_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=0.5,
        ),
        conv_cfg=dict(type='Conv3d', bias=False),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        class_frequencies=class_frequencies,
        class_weight_multipliers=class_weight_multipliers,
        focal_gamma=2.0,
    ),
)

learning_rate = 3e-4
training_steps = 25000

# Full validation every 1,000 global steps, including the 25k endpoint.
# None makes the interval continue across epoch boundaries.
val_check_interval = 1000
check_val_every_n_epoch = None

optimizer = dict(type='AdamW', lr=learning_rate, weight_decay=0.01)

lr_scheduler = dict(
    type='OneCycleLR',
    max_lr=learning_rate,
    total_steps=training_steps + 10,
    pct_start=0.05,
    cycle_momentum=False,
    anneal_strategy='cos',
    interval='step',
    frequency=1,
)

# Optional compatible image/depth initialization; not an OCC resume checkpoint.
load_from = os.environ.get('AERIALOCC_INIT_CHECKPOINT') or None
output_root = os.environ.get('AERIALOCC_OUTPUT_ROOT', 'outputs')
log_folder = 'aerialocc20k'
checkpoint_dirpath = os.path.join(output_root, log_folder, 'checkpoints')
checkpoint_every_n_train_steps = 500
checkpoint_save_top_k = 2
use_custom_collate = True
precision = 32
find_unused_parameters = False
gradient_clip_val = 5.0
save_path = None

# The teacher is training-only. Validation/inference never opens teacher maps.
test_pipeline[0]['load_teacher_depth'] = False
test_pipeline[-1]['meta_keys'].remove('vggt_teacher_depth')
testset_config['teacher_depth_root'] = None
if model['teacher_depth_loss_weight'] == 0:
    train_pipeline[0]['load_teacher_depth'] = False
    train_pipeline[-1]['meta_keys'].remove('vggt_teacher_depth')
    trainset_config['teacher_depth_root'] = None
del os
