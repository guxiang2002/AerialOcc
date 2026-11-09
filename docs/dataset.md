# AerialOcc-20K

Download the dataset from
[ModelScope](https://www.modelscope.cn/datasets/HHUguxiang/AerialOcc-20K), then
point `AERIALOCC_DATA_ROOT` to the extracted directory containing `Train/` and
`Val/`. The split uses 16,000 training frames and 4,000 validation frames.
Datasets, caches and trained weights are not stored in this Git repository.

```bash
export AERIALOCC_DATA_ROOT=/path/to/AerialOcc-20K
```

## Directory layout expected by the adapter

```text
AerialOcc-20K/
  Train/scene_00/cartesian/
    bev/000000.png
    velodyne/000000.bin
    labels/000000.label
    evaluation/params.json
    evaluation/000000.label
    evaluation/000000.bin
    poses.txt
  Val/scene_16/cartesian/...
  preprocess/labels/scene_00/000000_1_1.npy
  depth/sequences/scene_00/000000.npy
  vggt_depth/sequences/scene_00/000000.npy
```

Scene/frame names above are examples; the adapter scans the files actually
present. LiDAR files contain float32 XYZI. Evaluation `.label` files contain
occupancy labels and evaluation `.bin` files contain voxel observation counts.
`params.json` defines each scene's native grid. Preprocessing preserves unknown
cells as 255 and remaps native grids to the common model grid. The classifier
has 27 channels: empty plus 26 foreground classes. The fixed raw-to-compact map
and training frequencies are in `configs/aerialocc20k.py`.

## Occupancy labels and metric input depth

Run these only if their prepared outputs are absent from your downloaded data:

```bash
python tools/aerialocc_prepare_labels.py --data-root "$AERIALOCC_DATA_ROOT" --output-root "$AERIALOCC_DATA_ROOT/preprocess"
python tools/aerialocc_prepare_depth.py --data-root "$AERIALOCC_DATA_ROOT" --output-root "$AERIALOCC_DATA_ROOT/depth"
```

The depth tool projects supplied LiDAR into each aerial image. This input
preparation is separate from the VGGT teacher. Cached input depth is used at
both training and inference; the pipeline can also project the supplied point
cloud when an input depth cache is absent. This preset is not RGB-only.

## Frozen VGGT training cache

Use the official VGGT checkout in a separate environment compatible with its
requirements. The exporter observes the real adjacent/current three images,
filters the lowest 20 percent of valid confidence values, median-normalizes
center-frame relative depth and saves it without reading occupancy GT.

```bash
python tools/aerialocc_prepare_teacher.py --dataset-root "$AERIALOCC_DATA_ROOT" --output-root "$AERIALOCC_DATA_ROOT/vggt_depth" --split train --window-size 3 --vggt-repo /path/to/vggt --checkpoint-path /path/to/VGGT-1B/model.pt
```

Teacher depth is relative, not metric. During training, the relative map is
scaled to a detached student-depth median. Teacher maps are required only when
KD is enabled. Validation/inference disable teacher loading.

## Configuration and checkpoint compatibility

The single preset `configs/aerialocc20k.py` uses the retained temporal student,
three RGB frames `[-1,0,+1]` at 384x704, input depth, the 256x256x32 grid and
26 foreground classes. Defaults use CE/Focal/semantic/geometric scaling weights
1/1/1/0.5, relative VGGT KL weight 0.1 with internal coefficient 1, and directional
geometry weight 1. Geometry selects compact car ID 13 (raw ID 14). No learned
detection head is added. Geometry mapping and teacher supervision are training-only.

```bash
export AERIALOCC_OUTPUT_ROOT=outputs
export AERIALOCC_KD_WEIGHT=0.1
export AERIALOCC_GEOMETRY_WEIGHT=1.0
```

Settings must match the checkpoint being restored. For a legacy temporal OCC
checkpoint without the auxiliary geometry parameters, set
`AERIALOCC_GEOMETRY_WEIGHT=0` and its original KD setting. The exporter loads
strictly and does not silently discard learned tensors. This portable preset
is not a claim to reproduce a historical experiment merely by matching steps.
