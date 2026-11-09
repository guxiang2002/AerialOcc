"""Export OCC labels and real winning probabilities for the CPU26-class tools.

This is an explicit inference command, never called by repository import/tests.
Inputs and outputs use the released AerialOcc-20K 26-class label space.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    import misc
    import numpy as np
    import torch
    import mmdet3d_plugin
    from mmcv import Config
    from mmdet.datasets import build_dataset
    from torch.utils.data import DataLoader
    from LightningTools.dataset_dm import flexible_collate
    from LightningTools.pl_model import pl_model
    cfg = Config.fromfile(a.config)
    cfg.load_from = None
    cfg.pretrain = False
    cfg.eval = True
    from aerialocc.training.checkpoint import disable_external_initialization
    disable_external_initialization(cfg.model)
    model = pl_model(cfg)
    checkpoint = torch.load(a.checkpoint, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    if all(k.startswith('model.') for k in state):
        model.load_state_dict(state, strict=True)
    else:
        model.model.load_state_dict(state, strict=True)
    net = model.model.cuda().eval()
    if not torch.cuda.is_available():
        raise RuntimeError('OCC inference requires the configured CUDA extensions and a CUDA device.')
    dataset = build_dataset(cfg.data.test)
    loader = DataLoader(dataset, batch_size=1, num_workers=a.workers,
                        shuffle=False, pin_memory=True, collate_fn=flexible_collate)
    captured, frames = [], []

    def move(v):
        if torch.is_tensor(v): return v.cuda(non_blocking=True)
        if isinstance(v, dict): return {k: move(x) for k, x in v.items()}
        if isinstance(v, tuple): return tuple(move(x) for x in v)
        if isinstance(v, list): return [move(x) for x in v]
        return v

    def capture(head, inputs, output):
        captured.append(output['output_voxels'])

    def text(value):
        return str(value[0] if isinstance(value, (list, tuple)) else value)

    handle = net.pts_bbox_head.register_forward_hook(capture)
    try:
        for index, batch in enumerate(loader):
            if a.max_frames is not None and index >= a.max_frames:
                break
            gt = batch['gt_occ'].cpu().numpy()[0].astype(np.uint8)
            batch = move(batch)
            batch['gt_occ'] = None
            meta = batch['img_metas']
            meta.pop('vggt_teacher_depth', None)
            meta.pop('gt_depths', None)
            token = text(meta['sequence'])+'/'+text(meta['frame_id'])
            rel = Path(token)
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe frame token: '+token)
            with torch.no_grad():
                out = net(batch)
                if len(captured) != 1:
                    raise RuntimeError('Expected one occupancy-logit output per frame.')
                logits = captured.pop()
                if logits.shape[1] != 27 or not torch.isfinite(logits).all():
                    raise ValueError('26-class export requires27 finite output channels.')
                pred = out['pred'].long()
                winning = logits.float().softmax(1).gather(1, pred[:, None]).squeeze(1)
            pp = a.output/'pred'/rel.with_suffix('.npz')
            gp = a.output/'gt'/rel.with_suffix('.npy')
            pp.parent.mkdir(parents=True, exist_ok=True)
            gp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(pp, pred=pred[0].cpu().numpy().astype(np.uint8),
                                winning_probability=winning[0].cpu().numpy().astype(np.float32))
            np.save(gp, gt)
            frames.append(dict(token=token, pred=str(pp.relative_to(a.output)).replace('\\', '/'),
                               gt=str(gp.relative_to(a.output)).replace('\\', '/')))
            del logits, out, winning, pred, batch
    finally:
        handle.remove()
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'frames.json').write_text(json.dumps(frames, indent=2), encoding='utf-8')
    print('[AerialOcc] exported %d frames; cached GT is contiguous0..26/255, use --gt-label-space contiguous' % len(frames))


if __name__ == '__main__':
    main()
