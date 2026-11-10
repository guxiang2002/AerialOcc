# AerialOcc-20K checkpoint

The current release is the native 25,000-step checkpoint of
`configs/aerialocc20k.py`. It replaces the previous temporal-rebuild Model Zoo
entry and includes the full temporal student and training-only geometric
auxiliary parameters. All 1,876 state entries strictly load into the public
configuration and were verified tensor-for-tensor.

- Download: [best.ckpt](https://www.modelscope.cn/api/v1/models/HHUguxiang/AerialOcc/repo?Revision=master&FilePath=aerialocc20k%2Fbest.ckpt)
- Size: 1,915,028,100 bytes.
- SHA-256: `8a894a6339d30d0bde91a1aea13d47b2d3a84eb3eee631c7a6c04fde97804d03`.
- Native optimizer and scheduler states are retained; the checkpoint contains
  25,000 actual optimizer updates. The published bytes are unchanged.
- Inputs: three-frame RGB at 704 x 384 plus cached sensor-provided depth.
- Inference uses the temporal student; the VGGT teacher is training-only.

## Validation and selection

The released run used the original epoch-end validation schedule at 8k, 16k and
24k. Its retained 24k checkpoint was selected by 26-class SSC mIoU. A separate
full validation of final25k was better on both OCC IoU and SSC mIoU, so final25k
is now published as `best.ckpt`. This identifies the best available evaluated
OCC checkpoint; it does not assert a comparison with unretained 1k milestones.

For future runs, the current configuration validates the complete validation
split every 1,000 global steps, including the 25k endpoint, and updates `best`
by SSC mIoU. This cadence update does not change the released model tensors.

## Full validation results

Both checkpoints below were evaluated on the same 4,000 validation frames using
RTX4090D GPUs and FP32. Numbers are percentages.

| Checkpoint | OCC IoU | Precision | Recall | 26-class SSC mIoU | Occupied-voxel SEG mIoU | LiDAR-point SEG mIoU |
| --- | --- | --- | --- | --- | --- | --- |
| Retained 24k | 68.8372 | 80.8029 | 82.2961 | 29.7391 | 34.5196 | 14.4481 |
| Released 25k | 69.0187 | 81.1054 | 82.2423 | 29.9288 | 34.7511 | 14.5058 |

Occupied-voxel SEG and LiDAR-point SEG use different evaluation populations and
are reported separately.

| Released25k, no GT-known assistance | Center-distance AP (%) | 3D AABB AP@0.25 (%) | 3D AABB AP@0.50 (%) | mATE_XY (m) | mASE |
| --- | --- | --- | --- | --- | --- |
| car | 30.9358 | 37.6125 | 33.8736 | 0.0380 | 0.2071 |
| Fixed 26-class macro | 3.5898 | 5.2664 | 1.8519 | 0.7735 | 0.7962 |

Detection is evaluated against fixed semantic-region AABBs, not independently
annotated actor instances or oriented-box ground truth. GT-assisted diagnostics
are separate from the unassisted results above.
