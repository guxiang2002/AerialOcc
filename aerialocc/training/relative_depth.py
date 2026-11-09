import torch
import torch.nn.functional as F
from mmdet3d.models.builder import NECKS
from mmdet3d_plugin.models.img2bev.GeometryDepth_NetCarla import GeometryDepth_NetCarla
@NECKS.register_module()
class AerialOccRelativeDepth(GeometryDepth_NetCarla):
    """Teacher-relative shape supervision; no assumed metric VGGT scale.

    Scale each teacher map to the detached student median during training.
    Absolute scale remains constrained by metric occupancy and calibration.
    """
    def get_depth_loss(self,depth_labels,depth_preds):
        with torch.cuda.amp.autocast(enabled=False):
            p=depth_preds.float()
            if p.dim()==5:p=p.flatten(0,1)
            d=depth_labels.float().flatten(0,1)
            valid=torch.isfinite(d)&(d>0)
            log_d=torch.where(valid,d.clamp_min(1e-6).log(),torch.zeros_like(d))
            weight=F.adaptive_avg_pool2d(valid.float().unsqueeze(1),p.shape[-2:])[:,0]
            t=(F.adaptive_avg_pool2d(log_d.unsqueeze(1),p.shape[-2:])[:,0]/weight.clamp_min(1e-6)).exp()
            valid=weight>.5
            bins=torch.arange(*self.grid_config['dbound'],device=p.device,dtype=torch.float32)
            assert bins.numel()==p.shape[1]
            expected=(p.detach()*bins[None,:,None,None]).sum(1)
            scales=[]
            for i in range(len(t)):
                if int(valid[i].sum())<16:raise ValueError('Insufficient confident VGGT supervision')
                s=expected[i][valid[i]].median()/t[i][valid[i]].median().clamp_min(1e-6)
                scales.append(s)
            target=t*torch.stack(scales)[:,None,None]
            target=target.clamp(float(bins[0]),float(bins[-1]))
            sigma=(target*.05).clamp_min(.5)
            q=(-.5*((bins[None,:,None,None]-target[:,None])/sigma[:,None])**2).softmax(1).detach()
            kl=(q*(q.clamp_min(1e-8).log()-p.clamp_min(1e-8).log())).sum(1)
            loss=kl[valid].mean()*self.loss_depth_weight
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite VGGT KL')
            self.last_teacher_diagnostic=dict(valid_pixels=int(valid.sum()),
                unweighted_KL=float(loss.detach()),scale_min=float(torch.stack(scales).min()),
                scale_max=float(torch.stack(scales).max()))
            return loss

