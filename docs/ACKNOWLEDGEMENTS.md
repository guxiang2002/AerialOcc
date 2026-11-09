# Source acknowledgements

AerialOcc builds on [CGFormer](https://github.com/pkqbajng/CGFormer), under the
retained [Apache-2.0 license](../LICENSE). Source import and repository naming do
not transfer authorship of upstream code. Existing source copyright notices
remain intact. The original citation is:

```bibtex
@inproceedings{CGFormer,
 title = {Context and Geometry Aware Voxel Transformer for Semantic Scene Completion},
 author = {Yu, Zhu and Zhang, Runmin and Ying, Jiacheng and Yu, Junchen and Hu, Xiaohai and Luo, Lun and Cao, Si-Yuan and Shen, Hui-liang},
 booktitle = {Advances in Neural Information Processing Systems},
 pages = {1531--1555},
 volume = {37},
 year = {2024}
}
```

The unchanged MMDetection3D runtime is installed from upstream CGFormer commit
`2efd2c70cb2fe9e89a206f2ee6d3a62bae85f891` by `packages/setup.sh`.
Its `packages/mmdetection3d` tree hash is
`c018afa1f76b1880437bcc3c8628f23dd11749cc`, exactly the previously vendored tree.
Its unrelated dataset adapters and examples are not republished here.

We also acknowledge [MMDetection3D](https://github.com/open-mmlab/mmdetection3d),
[VoxFormer](https://github.com/NVlabs/VoxFormer),
[DFA3D](https://github.com/IDEA-Research/3D-deformable-attention), and
[VGGT](https://github.com/facebookresearch/vggt).
VGGT is an external frozen training teacher and retains its own license.
