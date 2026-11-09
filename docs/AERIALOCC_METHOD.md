# AerialOcc-20K implementation

## Temporal student

`mmdet3d_plugin/datasets/aerialocc20k.py` selects real `[-1,0,+1]` frames within
each scene. Missing neighbors at boundaries are replicated from the center.
`TemporalFeatureFusion` combines image features before depth-aware voxel
lifting, the transformer, encoder and occupancy head. The released preset does
not enable optional pose warping. Its input depth is cached or projected from
the supplied LiDAR at both training and inference.

## Training-only geometric supervision

`aerialocc/training/relative_depth.py` retains the teacher-relative KL formula:
per-image teacher scale is aligned to a detached student-depth median, converted
to a Gaussian target over depth bins and compared with the student's depth
distribution. The preset uses internal coefficient 1 and default outer weight
0.1. Frozen VGGT produces offline training maps and is not required by inference.

`aerialocc/training/geometry_loss.py` retains normalized same-semantic run-length
targets along positive/negative X/Y/Z, with unknown/grid-edge terminations
censored. A masked L1 loss trains the auxiliary mapping from shared voxel
features. This semantic-boundary target is not actor-ID or independent-instance
box supervision. For this dataset, compact car ID 13 corresponds to raw ID 14.
No learned detection head is introduced.

The registered `AerialOcc` wrapper connects these existing loss implementations
in `mmdet3d_plugin/models/detectors/AerialOcc.py`. `CGFormer` remains the internal
base class for inherited layers/checkpoint keys. The backbone, temporal fusion,
voxel transformer, encoder and occupancy decoder are retained.

## Inference and downstream tasks

Inference strips occupancy GT, teacher and direct GT-depth metadata before the
predictor is called. An evaluator may still load GT separately for scoring.
The geometry auxiliary is not called during inference. A single occupancy
output supports occupied-voxel segmentation, optional point projection and
component-to-box detection. See [evaluation definitions](EVALUATION_26.md).
Other dataset adapters, historical runs and experiment queues are not released.
