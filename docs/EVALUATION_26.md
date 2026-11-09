# 26-class evaluation after OCC inference

Run the commands in the root README **after** model inference. This CPU-only
tool never invokes the model or supplies GT to it. It does not regenerate depth.
Install `numpy scipy opencv-python`; torch/MMDet are not needed for this stage.

For compatible 27-channel AerialOcc-20K checkpoints, a bridge exporter is supplied:

```bash
python tools/aerialocc_export_occ.py --config configs/aerialocc20k.py --checkpoint /path/to/model.ckpt --output /path/to/export
python tools/aerialocc_eval26.py --manifest /path/to/export/frames.json --gt-label-space contiguous --task all --output /path/to/export/results26.json
```

The exporter strictly loads model weights, sets GT=None and removes teacher/GT
depth metadata before forwarding. It saves evaluation GT separately, never into
the predictor. It uses FP32 inference (not an efficiency benchmark). Dataset
preprocessing may load GT for the separate scoring cache. Existing experiment
evaluators should be used when reproducing their exact AMP/runtime protocol.

## Manifest

Use a JSON list with one entry per unique center frame. Relative paths resolve
against the manifest directory:

```json
[
  {"token": "scene_16/000000", "pred": "pred/scene_16/000000.npz", "gt": "gt/scene_16/000000_1_1.npy"}
]
```

Prediction NPZ: `pred`, integer `[X,Y,Z]`, **contiguous0..26**, empty0; optionally
`winning_probability`, float32 of the same shape. These probabilities are the
softmax probability of the argmax class at each voxel, not max logits or binary
occupancy probabilities. Detection requires them; no uniform artificial score
is substituted. A separate NPZ path can be set with the manifest's `confidence`.
Original logits need not be stored. NPY predictions also work for OCC/SEG.

GT defaults to the existing CARLA raw mapping:
`[0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28]`
maps to contiguous0..26; all other IDs map to255. Use
`--gt-label-space contiguous` if your GT has already been remapped. Predictions
must never be remapped twice. Class order is in `eval26.py`; this is the fixed AerialOcc-20K label order.

Defaults: grid256×256×32, origin(-25.6,-25.6,-32), step(.3,.2,1.25) meters,
fixed detection ROI first170×256×32 cells. This is the released anisotropic aerial grid. Override `--grid-config grid.json` when appropriate:

```json
{"shape":[256,256,32],"origin":[-25.6,-25.6,-32.0],"step":[0.3,0.2,1.25],"roi_shape":[170,256,32]}
```

## Metrics

The27×27 confusion matrix ignores GT255, includes empty0, and yields:

* OCC IoU/precision/recall: occupied vs empty on valid GT cells.
* SSC mIoU: fixed mean over26 foreground classes; empty is excluded from the
  mean but its false positives/negatives remain in each class's denominator.
* SEG mIoU/mAcc/OA: zero only the **GT-empty row**, keep the predicted-empty
  column. This evaluates semantic labels at GT-occupied voxels; it is not point
  segmentation. Empty predictions remain errors.

Optional point SEG: add `"points":"points/frame.npz"`, arrays `xyz:[N,3]`
and `labels:[N]`. XYZ must already be in the same metric frame as the grid;
labels use the selected GT label space. Points inside the grid are sampled by
floor voxel indexing, with empty predictions counted as errors. The result
stores a separate point confusion matrix. Points outside the prediction grid
are excluded by this declared ROI policy; it is not full-cloud coverage.

## Detection readout and scoring

The GT-free prediction path uses classwise26-neighbor3D connected components,
minimum8 predicted voxels, complete-cell XY minimum-area rectangle and full Z
extent, then converts the oriented generated box to its enclosing AABB. Scores
are the mean of the top20% winning-class probabilities in each component.
The source formula matches the existing complete-cell M2B implementation.
No teacher, GT mask or GT instance box is accepted by this raw readout.

GT boxes are the classwise26-neighbor **semantic-region** AABBs within the fixed
ROI, retaining singleton regions. They are pseudo-instances derived from
semantic voxels, not independent actor-instance annotations.

Same-class predictions are ranked by descending confidence and greedily
matched one-to-one. Center AP uses XY thresholds0.5/1/2/4m, strict `<`, followed
by the inherited101-point interpolation/min-recall0.1/min-precision0.1 rule.
Center mAP averages those four APs and26 classes.3D AP@.25/.50 and BEV AP@.50
use `>=` IoU thresholds and all-points interpolated AP. All means use fixed26
classes; absent classes have AP0 with `supported=false` disclosed. This is not
an official autonomous-driving leaderboard evaluator, and no NDS is inferred from
incomplete velocity/attribute/yaw labels.

Default predictions are clipped only by the declared geometric ROI. With
`--gt-known-diagnostic`, an additional result clips the same proposals using
the binary GT valid domain (no class lookup), retaining each proposal's class
and score. This is **evaluation-time GT assistance**, not deployable inference.
Do not put its AP in a GT-free main table.

NPZ/JSON output keys clearly identify AABB scores; do not call them oriented
AP. JSON metrics are fractions0..1; the accompanying CSV is percent. Keep the
grid, class map, readout, checkpoints and GT protocol fixed across methods.

The evaluator is a portable extraction of existing formulas. Historical
EPoC/RGB-token grouping experiments use a different readout and must be labeled
separately; this pure3DCC tool does not silently reproduce their scores.
