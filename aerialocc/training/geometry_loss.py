"""VoxNT-inspired six-direction supervision with censored unknown boundaries.

This is a training-only adaptation, not the complete VoxDet architecture.
Offsets count the current voxel, matching the run-length target convention.
Unknown cells and rays terminated by unknown/the grid edge are not supervised.
"""
import torch
from torch import nn
import torch.nn.functional as F


def directional_target(labels, axis, positive, ignore_index=255):
    """Return normalized run lengths and a mask of observed class boundaries."""
    t=labels.transpose(axis+1,-1)
    if positive:t=t.flip([-1])
    length=t.shape[-1]
    index=torch.arange(length,device=t.device,dtype=torch.long)
    index=index.reshape((1,)*(t.dim()-1)+(length,)).expand_as(t)
    changed=torch.ones_like(t,dtype=torch.bool)
    changed[...,1:]=t[...,1:]!=t[...,:-1]
    start=torch.where(changed,index,torch.zeros_like(index)).cummax(-1)[0]
    boundary=t.gather(-1,(start-1).clamp_min(0))
    valid=(start>0)&(boundary!=ignore_index)&(t!=ignore_index)
    target=(index-start+1).float()/float(length)
    if positive:target=target.flip([-1]);valid=valid.flip([-1])
    return target.transpose(axis+1,-1),valid.transpose(axis+1,-1)


@torch.no_grad()
def geometry_targets(labels,shape,class_ids=(13,),ignore_index=255):
    assert labels.dim()==4
    if tuple(labels.shape[1:])!=tuple(shape):
        labels=F.interpolate(labels[:,None].float(),size=shape,mode='nearest')[:,0].long()
    selected=torch.zeros_like(labels,dtype=torch.bool)
    for category in class_ids:selected|=labels==int(category)
    targets=[];masks=[]
    for axis in range(3):
        for positive in [True,False]:
            target,valid=directional_target(labels,axis,positive,ignore_index)
            targets.append(target);masks.append(valid&selected)
    return torch.stack(targets,1),torch.stack(masks,1)


class InstanceGeometryAuxiliary(nn.Module):
    def __init__(self,in_channels=128,hidden_channels=32,class_ids=(13,),init_seed=7241):
        super().__init__()
        self.class_ids=tuple(class_ids)
        # Auxiliary initialization must not advance the shared-model/data RNG.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(init_seed)
            self.predictor=nn.Sequential(nn.Conv3d(in_channels,hidden_channels,1),
                nn.GroupNorm(8,hidden_channels),nn.ReLU(inplace=True),nn.Conv3d(hidden_channels,6,1))

    def forward(self,features,labels):
        logits=self.predictor(features)
        with torch.cuda.amp.autocast(enabled=False):
            predicted=logits.float().sigmoid()
            target,mask=geometry_targets(labels,predicted.shape[2:],self.class_ids)
            denominator=mask.sum().clamp_min(1)
            loss=((predicted-target).abs()*mask).sum()/denominator
        # Detached tensors avoid a GPU synchronization on each microbatch.
        diagnostic={'valid_direction_targets':mask.sum().detach(),
                    'unweighted_loss':loss.detach()}
        return loss,diagnostic
