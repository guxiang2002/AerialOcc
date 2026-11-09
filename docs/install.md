# AerialOcc-20K installation

The retained runtime uses Linux, an NVIDIA GPU, a CUDA toolkit/compiler, Python
3.7 and the PyTorch 1.10.1/CUDA 11.3 stack. This is the existing tested runtime;
newer OpenMMLab major versions are not interchangeable with it.

```bash
conda create -n aerialocc python=3.7 -y
conda activate aerialocc
python -m pip install pip==23.3.2 setuptools==59.5.0 wheel
python -m pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
python -m pip install openmim
mim install mmcv-full==1.4.0
python -m pip install mmdet==2.14.0 mmsegmentation==0.14.1
python -m pip install -r docs/requirements.txt
python -m pip install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels
bash packages/setup.sh
```

`packages/setup.sh` installs the unchanged upstream MMDetection3D runtime from a
fixed commit under ignored `.deps/`, then builds the supplied DFA3D extension.
It downloads no dataset or weights. Build extensions for your Python/PyTorch/
CUDA stack; compiled `.so` files from another machine are not distributed.
On GPUs needing a newer compiler, use a toolkit compatible with your local
extension build and PyTorch runtime.

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

Trained AerialOcc weights are not included. Data and training-only VGGT-cache
preparation are in [dataset.md](dataset.md). VGGT uses a separate environment
following its official requirements. CPU-only exported-result evaluation needs
NumPy/SciPy/OpenCV, without the model runtime. Optional visualization dependencies
are in [visualization.md](visualization.md).
