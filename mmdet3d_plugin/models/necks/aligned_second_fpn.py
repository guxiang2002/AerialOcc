import torch
import torch.nn.functional as F
from mmdet.models import NECKS
from mmdet3d.models.necks.second_fpn import SECONDFPN


@NECKS.register_module()
class AlignedSECONDFPN(SECONDFPN):
    """SECONDFPN with a final spatial alignment before concatenation.

    EfficientNet can produce one-pixel scale mismatches for image sizes such as
    450x800. The regular SECONDFPN concatenates feature maps directly, so we
    align all deblock outputs to the first output's HxW.
    """

    def forward(self, x):
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        if len(ups) > 1:
            target_size = ups[0].shape[-2:]
            ups = [
                up
                if up.shape[-2:] == target_size
                else F.interpolate(
                    up,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False,
                )
                for up in ups
            ]
            out = torch.cat(ups, dim=1)
        else:
            out = ups[0]
        return [out]
