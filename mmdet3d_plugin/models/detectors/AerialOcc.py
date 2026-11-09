"""AerialOcc-20K temporal student and its training-only geometric mapping."""
from mmdet.models import DETECTORS

from .CGFormer import CGFormer
from aerialocc.training.geometry_loss import InstanceGeometryAuxiliary
from aerialocc.training.relative_depth import AerialOccRelativeDepth  # registry


@DETECTORS.register_module()
class AerialOcc(CGFormer):
    def __init__(self, instance_geometry_weight=1.0,
                 geometry_class_ids=(13,), **kwargs):
        super().__init__(**kwargs)
        # The base image encoder uses MMCV's lazy initialization interface.
        self.img_backbone.init_weights()
        self.instance_geometry_weight = float(instance_geometry_weight)
        if self.instance_geometry_weight < 0:
            raise ValueError('instance_geometry_weight must be nonnegative')
        if self.instance_geometry_weight:
            self.train_only_geometry = InstanceGeometryAuxiliary(
                class_ids=geometry_class_ids)
        self._geometry_latent = None
        self.last_geometry_diagnostic = {}

    def occ_encoder(self, features):
        output = super().occ_encoder(features)
        if self.training and self.instance_geometry_weight:
            self._geometry_latent = (
                output[0] if isinstance(output, (tuple, list)) else output)
        return output

    def forward_train(self, data_dict):
        if (self.teacher_depth_loss_weight > 0 and
                'vggt_teacher_depth' not in data_dict['img_metas']):
            raise ValueError('Training with KD requires the frozen VGGT cache; '
                             'run tools/aerialocc_prepare_teacher.py first.')
        output = super().forward_train(data_dict)
        if self.instance_geometry_weight:
            features, self._geometry_latent = self._geometry_latent, None
            loss, diagnostic = self.train_only_geometry(
                features, data_dict['gt_occ'])
            output['losses']['loss_instance_geometry'] = (
                self.instance_geometry_weight * loss)
            self.last_geometry_diagnostic = diagnostic
        return output

    def forward_test(self, data_dict):
        # GT belongs to scoring only. The teacher and direct GT depth never
        # enter the predictor, even when evaluation loads annotations.
        target = data_dict.get('gt_occ')
        inputs = dict(data_dict)
        inputs['gt_occ'] = None
        inputs['img_metas'] = dict(data_dict['img_metas'])
        for key in ('vggt_teacher_depth', 'gt_depths', 'gt_semantics'):
            inputs['img_metas'].pop(key, None)
        output = super().forward_test(inputs)
        output['gt_occ'] = target
        return output

    def remove_training_branches(self):
        if hasattr(self, 'train_only_geometry'):
            del self.train_only_geometry
        self._geometry_latent = None
