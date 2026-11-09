# AerialOcc-20K visualization

Install the optional renderer in a Mayavi/VTK-compatible environment:

```bash
python -m pip install mayavi vtk PyQt5 configobj
```

First save the predictions using the checkpoint/configuration used for scoring:

```bash
python main.py --config_path configs/aerialocc20k.py --eval --ckpt_path /path/to/model.ckpt --save_path outputs/predictions
```

The preset saves original raw IDs for the existing aerial color palette. Select
an actual validation scene and frame from your downloaded data:

```bash
python tools/aerialocc_visualize.py --data-root "$AERIALOCC_DATA_ROOT" --pred-root outputs/predictions --scene scene_16 --frame 000000 --output-root outputs/visualizations
```

Use `--view topdown`, `--view side` or `--view lane_oblique` for camera presets.
Run `python tools/aerialocc_visualize.py --help` for all options. The offscreen
Mayavi renderer may require EGL/X11/a virtual display on headless Linux.
Use `--source gt_model` to render model-grid GT separately. Empty/unknown cells
are hidden; rendering does not change the saved predictions or evaluation.
