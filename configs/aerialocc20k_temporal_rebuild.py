"""Checkpoint-compatible configuration for the released temporal-rebuild run.

The default full-method configuration has an additional training-only geometry
branch and corrected relative-depth KL. These legacy weights do not include it.
"""
import os

_base_ = './aerialocc20k.py'

# Retain AerialOcc's display name and GT-independent inference wrapper, with
# exactly the same parameter keys/shapes as the historical temporal student.
model = dict(
    instance_geometry_weight=0.0,
    teacher_depth_loss_weight=0.1,
    depth_net=dict(
        type='GeometryDepth_NetCarla',
        loss_depth_weight=0.0001,
    ),
)

gradient_clip_val = 0.0
checkpoint_save_top_k = -1
log_folder = 'aerialocc20k_temporal_rebuild'
checkpoint_dirpath = os.path.join(
    os.environ.get('AERIALOCC_OUTPUT_ROOT', 'outputs'), log_folder, 'checkpoints')
del os
