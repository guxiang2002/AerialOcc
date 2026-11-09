# Step-by-step installation instructions

AerialOcc is built on [CGFormer](https://github.com/pkqbajng/CGFormer) and the
OccFormer codebase. These instructions follow the upstream
[CGFormer installation guide](https://github.com/pkqbajng/CGFormer/blob/main/docs/install.md),
with this repository's dependency setup command.

**a. Create a conda virtual environment and activate it.**

The retained runtime uses Python 3.7; use the pinned stack below rather than
substituting newer OpenMMLab major versions.

```bash
conda create -n aerialocc python=3.7 -y
conda activate aerialocc
python -m pip install pip==23.3.2 setuptools==59.5.0 wheel
```

**b. Install PyTorch and torchvision following the official instructions.**

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

Or install with pip:

```bash
python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

The original MMDetection3D 0.17.1 runtime does not support PyTorch 1.11 or later.
CUDA extensions require an NVIDIA GPU and a local CUDA toolkit/compiler
compatible with the installed PyTorch stack.

**c. Install MMCV, MMDetection and MMSegmentation.**

```bash
python -m pip install openmim
mim install mmcv-full==1.4.0
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**d. Install the MMDetection3D runtime, DFA3D and remaining requirements.**

Run these commands from the AerialOcc repository root:

```bash
bash packages/setup.sh
python -m pip install -r docs/requirements.txt
```

`packages/setup.sh` fetches the unchanged upstream runtime from a fixed commit
into ignored `.deps/`, then builds the supplied DFA3D extension. It downloads no
dataset or weights. Build extensions for your own Python/PyTorch/CUDA stack;
compiled `.so` files from another machine are not distributed.

**e. Install the upstream compatibility fixes.**

```bash
python -m pip install yapf==0.40.0
python -m pip install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels
```

## Initialization weights

Supply the EfficientNet-B7 ImageNet and Swin-T initialization files used by the
retained architecture. Their model zoos are
[MMClassification](https://github.com/open-mmlab/mmclassification/tree/v0.23.2/configs/efficientnet)
and [Swin Transformer](https://github.com/microsoft/Swin-Transformer).
Set explicit local paths before training:

```bash
export AERIALOCC_B7_CHECKPOINT=/path/to/efficientnet-b7_3rdparty_8xb32-aa_in1k_20220119-bf03951c.pth
export AERIALOCC_SWIN_CHECKPOINT=/path/to/swin_tiny_patch4_window7_224.pth
```

Optionally set `AERIALOCC_INIT_CHECKPOINT=/path/to/initialization.pth` for a
compatible image/segmentation-depth initialization. This differs from
`main.py --ckpt_path`, which restores a complete training state or final model.
Initialization matching counts are printed; final inference checkpoints are
loaded strictly. Inference/restore disables external ImageNet weight loading,
so those initialization files are not required again.

Trained temporal-rebuild weights and their matching configuration are listed in the
[Model Zoo](../README.md#model-zoo). Data and training-only VGGT-cache
preparation are in [dataset.md](dataset.md). VGGT uses a separate environment
following its official requirements. CPU-only exported-result evaluation needs
NumPy/SciPy/OpenCV, without the model runtime. Optional visualization dependencies
are in [visualization.md](visualization.md).
