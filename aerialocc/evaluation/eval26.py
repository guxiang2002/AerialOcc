"""Export-only 26-class evaluation; no torch, checkpoints or model GT inputs.

Defaults match the historical CARLA-large anisotropic grid and valid ROI.
GT-derived boxes are semantic regions, not independently annotated instances.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

CLASSES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
           'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
           'pedestrian', 'rider', 'car', 'truck', 'bus', 'motorcycle',
           'bicycle', 'static', 'dynamic', 'other', 'water', 'road_line',
           'ground', 'bridge', 'rail_track', 'guard_rail']
RAW_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
METRICS = ['center_0.5', 'center_1.0', 'center_2.0', 'center_4.0',
           '3d_AP25', '3d_AP50', 'bev_AP50']
DEFAULT_GRID = dict(shape=[256, 256, 32], origin=[-25.6, -25.6, -32.0],
                    step=[0.3, 0.2, 1.25], roi_shape=[170, 256, 32])


def load_array(path, key):
    path = Path(path)
    if path.suffix == '.npy':
        return np.load(str(path), allow_pickle=False)
    with np.load(str(path), allow_pickle=False) as z:
        if key not in z:
            raise ValueError('%s has no %r array (available: %s)' %
                             (path, key, ', '.join(z.files)))
        return z[key].copy()


def labels27(array, space):
    array = np.asarray(array)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError('Labels must be integer IDs, not probabilities.')
    if space == 'raw':
        lut = np.full(256, 255, np.uint8)
        lut[RAW_IDS] = np.arange(27, dtype=np.uint8)
        if array.min() < 0 or array.max() > 255:
            raise ValueError('Raw IDs outside uint8 range.')
        return lut[array]
    if not np.all(((array >= 0) & (array <= 26)) | (array == 255)):
        raise ValueError('Expected contiguous0..26 or ignore255.')
    return array.astype(np.uint8)


def confusion(pred, gt):
    valid = gt != 255
    if pred.shape != gt.shape or not np.all(pred[valid] <= 26):
        raise ValueError('Prediction/GT shape or valid label mismatch.')
    return np.bincount(27 * gt[valid].astype(np.int64) + pred[valid],
                       minlength=27 * 27).reshape(27, 27)


def semantic_metrics(cm, occupied_only=False):
    cm = np.asarray(cm, np.int64).copy()
    if occupied_only:
        cm[0, :] = 0  # Keep the predicted-empty column: those are false negatives.
    tp = cm.diagonal().astype(float)
    actual, predicted = cm.sum(1), cm.sum(0)
    iou = tp / (actual + predicted - tp + 1e-5)
    acc = tp / (actual + 1e-5)
    hit = float(cm[1:, 1:].sum())
    fp, fn = float(cm[0, 1:].sum()), float(cm[1:, 0].sum())
    return dict(IoU=hit / max(hit + fp + fn, 1),
                Precision=hit / max(hit + fp, 1), Recall=hit / max(hit + fn, 1),
                mIoU=float(iou[1:].mean()), mAcc=float(acc[1:].mean()),
                OA=float(tp[1:].sum() / max(actual[1:].sum(), 1)) if occupied_only
                else float(tp.sum() / max(actual.sum(), 1)),
                per_class_IoU=dict(zip(CLASSES, iou[1:].tolist())),
                GT_counts=actual[1:].tolist(), fixed_macro_classes=26)


def oriented_box(coords, shape, origin, step):
    """Existing complete-cell M2B formula; bottom-centered9-vector."""
    import cv2
    xyz = origin + (coords + .5) * step
    xy = np.unique(xyz[:, :2], axis=0)
    corners = (xy[:, None, :] + np.array([[-.5, -.5], [-.5, .5],
               [.5, -.5], [.5, .5]])[None, :, :] * step[:2]).reshape(-1, 2)
    rectangle = cv2.boxPoints(cv2.minAreaRect(corners.astype(np.float32))).astype(np.float64)
    edge = rectangle[1] - rectangle[0]
    angle = math.atan2(edge[1], edge[0])
    c, s = math.cos(angle), math.sin(angle)
    basis = np.array([[c, -s], [s, c]])
    local = xyz[:, :2] @ basis
    padding = np.array([abs(c) * step[0] + abs(s) * step[1],
                        abs(s) * step[0] + abs(c) * step[1]])
    low, high = local.min(0) - padding / 2, local.max(0) + padding / 2
    center = (low + high) / 2 @ basis.T
    dx, dy = high - low
    if dy > dx:
        dx, dy = dy, dx
        angle += math.pi / 2
    angle = (angle + math.pi / 2) % math.pi - math.pi / 2
    bottom = origin[2] + coords[:, 2].min() * step[2]
    height = (coords[:, 2].max() - coords[:, 2].min() + 1) * step[2]
    return np.array([center[0], center[1], bottom, dx, dy, height, angle, 0, 0], np.float32)


def enclosing_aabb(box):
    x, y, z, dx, dy, dz, yaw = map(float, box[:7])
    c, s = abs(np.cos(yaw)), abs(np.sin(yaw))
    rx, ry = .5 * (c * dx + s * dy), .5 * (s * dx + c * dy)
    return np.array([x-rx, y-ry, z, x+rx, y+ry, z+dz], np.float64)


def component_boxes(pred, winning, grid, ground_truth=False):
    """GT-free predictions; singleton GT regions use an explicit separate mode."""
    origin, step = np.asarray(grid['origin']), np.asarray(grid['step'])
    p = pred[tuple(slice(0, n) for n in grid['roi_shape'])] if ground_truth else pred
    boxes, scores, labels = [], [], []
    for category in range(1, 27):
        mask = p == category
        if not mask.any():
            continue
        cc, _ = ndimage.label(mask, np.ones((3, 3, 3), np.uint8))
        coords = np.argwhere(mask)
        ids = cc[tuple(coords.T)]
        order = np.argsort(ids, kind='stable')
        boundaries = np.r_[0, np.flatnonzero(np.diff(ids[order])) + 1, len(order)]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if right-left < (1 if ground_truth else 8):
                continue
            c = coords[order[left:right]]
            if ground_truth:
                box = np.r_[origin+c.min(0)*step, origin+(c.max(0)+1)*step]
                score = 1.
            else:
                box = enclosing_aabb(oriented_box(c, pred.shape, origin, step))
                values = winning[tuple(c.T)]
                k = max(1, int(math.ceil(len(values)*.2)))
                score = float(np.partition(values, -k)[-k:].mean())
            boxes.append(box)
            scores.append(score)
            labels.append(category-1)
    return dict(boxes=np.asarray(boxes, np.float64).reshape(-1, 6),
                scores=np.asarray(scores, np.float64), labels=np.asarray(labels, np.int16))


def clip_box(box, grid, known=None):
    """Fixed ROI clip; known!=None is explicitly evaluation-time GT assistance."""
    origin, step = np.asarray(grid['origin']), np.asarray(grid['step'])
    roi = np.asarray(grid['roi_shape'])
    lo, hi = np.maximum(box[:3], origin), np.minimum(box[3:], origin+roi*step)
    if np.any(hi <= lo):
        return None
    if known is not None:
        lower = np.maximum(0, np.floor((lo-origin)/step+1e-7).astype(int))
        upper = np.minimum(roi, np.ceil((hi-origin)/step-1e-7).astype(int))
        if np.any(upper <= lower):
            return None
        crop = known[tuple(slice(l, h) for l, h in zip(lower, upper))]
        if not crop.any():
            return None
        mins, maxs = [], []
        for axis in range(3):
            ix = np.flatnonzero(crop.any(axis=tuple(a for a in range(3) if a != axis)))
            mins.append(lower[axis]+ix[0])
            maxs.append(lower[axis]+ix[-1]+1)
        lo, hi = np.maximum(lo, origin+np.asarray(mins)*step), np.minimum(hi, origin+np.asarray(maxs)*step)
    return np.r_[lo, hi] if np.all(hi > lo) else None


def matches(gt, pred, scores):
    matched = np.zeros((7, len(pred)), np.uint8)
    if not len(gt) or not len(pred):
        return matched
    gc, pc = .5*(gt[:, :3]+gt[:, 3:]), .5*(pred[:, :3]+pred[:, 3:])
    distance = np.linalg.norm(pc[:, None, :2]-gc[None, :, :2], axis=2)
    inter = np.maximum(np.minimum(pred[:, None, 3:], gt[None, :, 3:])-
                       np.maximum(pred[:, None, :3], gt[None, :, :3]), 0)
    volume = inter.prod(2)
    iou = volume / np.maximum((pred[:, 3:]-pred[:, :3]).prod(1)[:, None] +
                              (gt[:, 3:]-gt[:, :3]).prod(1)[None, :]-volume, 1e-12)
    area = inter[:, :, :2].prod(2)
    bev = area / np.maximum((pred[:, 3:5]-pred[:, :2]).prod(1)[:, None] +
                            (gt[:, 3:5]-gt[:, :2]).prod(1)[None, :]-area, 1e-12)
    order = np.argsort(scores, kind='stable')[::-1]
    for k, threshold in enumerate([.5, 1, 2, 4, .25, .5, .5]):
        used = np.zeros(len(gt), bool)
        for j in order:
            values = (distance if k < 4 else iou if k < 6 else bev)[j].copy()
            values[used] = np.inf if k < 4 else -1
            index = int(np.argmin(values) if k < 4 else np.argmax(values))
            ok = values[index] < threshold if k < 4 else values[index] >= threshold
            if ok:
                matched[k, j] = 1
                used[index] = True
    return matched


def average_precision(tp, count, center=False):
    if not count or not len(tp) or not np.any(tp):
        return 0.
    precision = np.cumsum(tp, dtype=float)/np.arange(1, len(tp)+1)
    recall = np.cumsum(tp, dtype=float)/count
    if center:
        interpolated = np.interp(np.linspace(0, 1, 101), recall, precision, right=0)
        return float(np.maximum(interpolated[11:]-.1, 0).mean()/.9)
    rec, pre = np.r_[0., recall, 1.], np.r_[0., precision, 0.]
    pre = np.maximum.accumulate(pre[::-1])[::-1]
    edges = np.flatnonzero(rec[1:] != rec[:-1])
    return float(np.sum((rec[edges+1]-rec[edges])*pre[edges+1]))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--task', choices=['occ-seg', 'det', 'all'], default='all')
    p.add_argument('--gt-label-space', choices=['raw', 'contiguous'], default='raw')
    p.add_argument('--pred-key', default='pred')
    p.add_argument('--gt-key', default='gt')
    p.add_argument('--grid-config', type=Path)
    p.add_argument('--gt-known-diagnostic', action='store_true')
    a = p.parse_args(argv)
    grid = json.loads(a.grid_config.read_text()) if a.grid_config else DEFAULT_GRID
    frames = json.loads(a.manifest.read_text())
    if not isinstance(frames, list) or not frames:
        raise ValueError('Manifest must be a nonempty JSON list.')
    base = a.manifest.resolve().parent
    resolve = lambda v: Path(v) if Path(v).is_absolute() else base/str(v)
    cm, pcm = np.zeros((27, 27), np.int64), np.zeros((27, 27), np.int64)
    modes = ['unmasked'] + (['gtknown_assisted'] if a.gt_known_diagnostic else [])
    buckets = {mode: [[] for _ in CLASSES] for mode in modes}
    counts, seen, records = np.zeros(26, np.int64), set(), []
    for frame in frames:
        token = frame['token']
        if token in seen:
            raise ValueError('Duplicate token: '+token)
        seen.add(token)
        pred = labels27(load_array(resolve(frame['pred']), a.pred_key), 'contiguous')
        gt = labels27(load_array(resolve(frame['gt']), a.gt_key), a.gt_label_space)
        if tuple(pred.shape) != tuple(grid['shape']) or pred.shape != gt.shape:
            raise ValueError('Grid mismatch at '+token)
        cm += confusion(pred, gt)
        if 'points' in frame:
            xyz = load_array(resolve(frame['points']), 'xyz')
            pg = labels27(load_array(resolve(frame['points']), 'labels'), a.gt_label_space)
            ix = np.floor((xyz-np.asarray(grid['origin']))/np.asarray(grid['step'])).astype(np.int64)
            inside = ((ix >= 0) & (ix < np.asarray(pred.shape))).all(1)
            pcm += confusion(pred[tuple(ix[inside].T)], pg[inside])
        if a.task == 'occ-seg':
            continue
        winning = load_array(resolve(frame.get('confidence', frame['pred'])), 'winning_probability')
        if winning.shape != pred.shape or not np.isfinite(winning).all() or not np.all((winning >= 0) & (winning <= 1)):
            raise ValueError('Provide dense, real winning-class softmax probabilities at '+token)
        raw = component_boxes(pred, winning, grid)
        # Labels/GT are first used by the scorer after prediction-to-box conversion.
        g = component_boxes(gt, None, grid, ground_truth=True)
        counts += np.bincount(g['labels'], minlength=26)
        for mode in modes:
            selected, boxes = [], []
            for j, box in enumerate(raw['boxes']):
                b = clip_box(box, grid, gt != 255 if mode == 'gtknown_assisted' else None)
                if b is not None:
                    boxes.append(b)
                    selected.append(j)
            boxes = np.asarray(boxes, np.float64).reshape(-1, 6)
            labels, scores = raw['labels'][selected], raw['scores'][selected]
            for category in range(26):
                select = labels == category
                tp = matches(g['boxes'][g['labels'] == category], boxes[select], scores[select])
                buckets[mode][category].append((scores[select], tp))
        records.append(dict(token=token,raw_predictions=len(raw['boxes']),GT_regions=len(g['boxes'])))
    result = dict(method='AerialOcc', frames=len(frames), classes=CLASSES, grid=grid,
                  metrics_scale='fractions0..1', OCC_SSC=semantic_metrics(cm),
                  SEG_GT_occupied_voxels=semantic_metrics(cm, True), confusion_matrix=cm.tolist(),
                  DET_definition='26-class semantic-region pseudo-GT AABB; not actor-instance or oriented-box AP',
                  GT_in_raw_readout=False, GT_known_assistance=a.gt_known_diagnostic)
    if pcm.sum():
        result['SEG_points_inside_grid'] = semantic_metrics(pcm, True)
        result['point_confusion_matrix'] = pcm.tolist()
    if a.task != 'occ-seg':
        result['DET'] = {}
        for mode in modes:
            per = []
            for category, name in enumerate(CLASSES):
                samples = buckets[mode][category]
                scores = np.concatenate([x[0] for x in samples])
                tp = np.concatenate([x[1] for x in samples], axis=1)
                order = np.argsort(scores, kind='stable')[::-1]
                values = [average_precision(tp[k, order], int(counts[category]), k < 4) for k in range(7)]
                per.append(dict(class_name=name, GT=int(counts[category]), predictions=len(scores),
                                supported=bool(counts[category]),
                                **dict(zip(METRICS, values))))
            result['DET'][mode] = dict(per_class=per, fixed_macro_classes=26,
                center_mAP=float(np.mean([[x[k] for k in METRICS[:4]] for x in per])),
                **{k:float(np.mean([x[k] for x in per])) for k in METRICS[4:]})
        result['frame_records'] = records
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    with a.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'percent'])
        for key in ['IoU', 'Precision', 'Recall', 'mIoU']:
            writer.writerow(['OCC_SSC/'+key, 100*result['OCC_SSC'][key]])
        for key in ['mIoU', 'mAcc', 'OA']:
            writer.writerow(['SEG_GT_occupied_voxels/'+key, 100*result['SEG_GT_occupied_voxels'][key]])
        for mode, values in result.get('DET', {}).items():
            for key in ['center_mAP']+METRICS[4:]:
                writer.writerow(['DET/'+mode+'/'+key, 100*values[key]])
    print('[AerialOcc] %d frames -> %s' % (len(frames), a.output))


if __name__ == '__main__':
    main()
