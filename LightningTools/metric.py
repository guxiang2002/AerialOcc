import numpy as np

try:
    from scipy import ndimage
except ImportError:
    ndimage = None


class OccupiedVoxelSemMetrics:
    """3D semantic segmentation metrics on labeled occupied elements.

    Unlike semantic scene completion, this metric ignores empty/free-space
    voxels by default and evaluates semantic classification only where the
    ground truth has an occupied semantic label.
    """

    def __init__(self, n_classes, empty_index=0, ignore_index=255):
        self.n_classes = int(n_classes)
        self.empty_index = int(empty_index)
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.n_classes, dtype=np.float64)
        self.fps = np.zeros(self.n_classes, dtype=np.float64)
        self.fns = np.zeros(self.n_classes, dtype=np.float64)
        self.gts = np.zeros(self.n_classes, dtype=np.float64)

    def add_batch(self, y_pred, y_true):
        y_pred = np.asarray(y_pred).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)

        valid = (y_true != self.ignore_index) & (y_true != self.empty_index)
        valid &= (y_true >= 0) & (y_true < self.n_classes)
        y_pred = y_pred[valid]
        y_true = y_true[valid]
        if y_true.size == 0:
            return

        pred_valid = (y_pred >= 0) & (y_pred < self.n_classes)
        y_pred = np.where(pred_valid, y_pred, -1)

        for class_id in range(self.n_classes):
            if class_id == self.empty_index:
                continue
            true_class = y_true == class_id
            pred_class = y_pred == class_id
            tp = np.logical_and(true_class, pred_class).sum()
            fp = np.logical_and(~true_class, pred_class).sum()
            fn = np.logical_and(true_class, ~pred_class).sum()
            self.tps[class_id] += tp
            self.fps[class_id] += fp
            self.fns[class_id] += fn
            self.gts[class_id] += true_class.sum()

    def get_stats(self):
        iou = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        acc = self.tps / (self.gts + 1e-5)
        valid_classes = np.ones(self.n_classes, dtype=bool)
        valid_classes[self.empty_index] = False
        valid_classes &= self.gts > 0
        occupied_mask = np.ones(self.n_classes, dtype=bool)
        occupied_mask[self.empty_index] = False
        occupied_total = float(np.sum(self.gts[occupied_mask]))
        occupied_correct = float(np.sum(self.tps[occupied_mask]))
        if np.any(valid_classes):
            miou = float(np.mean(iou[valid_classes]))
            macc = float(np.mean(acc[valid_classes]))
        else:
            miou = 0.0
            macc = 0.0
        return {
            'iou': iou,
            'acc': acc,
            'miou': miou,
            'macc': macc,
            'oa': occupied_correct / max(occupied_total, 1e-5),
            'valid_classes': valid_classes,
        }


class RemappedSemanticIoUMetrics:
    """Dense 3D semantic occupancy IoU after taxonomy remapping.

    Every valid voxel in the output grid contributes to TP/FP/FN. Class 0
    (free space) is evaluated normally, and only ``ignore_index`` voxels are
    excluded. This optional taxonomy diagnostic is separate from occupied-only
    semantic segmentation and point- or camera-visible-mask metrics.
    """

    def __init__(self, label_mapping, class_names, ignore_index=255, empty_index=None):
        self.label_mapping = np.asarray(label_mapping, dtype=np.int64)
        self.class_names = tuple(class_names)
        self.n_classes = len(self.class_names)
        self.ignore_index = int(ignore_index)
        self.empty_index = None if empty_index is None else int(empty_index)
        if self.label_mapping.ndim != 1:
            raise ValueError('label_mapping must be a one-dimensional sequence')
        if np.any((self.label_mapping < 0) | (self.label_mapping >= self.n_classes)):
            raise ValueError('label_mapping contains a target class outside class_names')
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.n_classes, dtype=np.float64)
        self.fps = np.zeros(self.n_classes, dtype=np.float64)
        self.fns = np.zeros(self.n_classes, dtype=np.float64)
        self.gts = np.zeros(self.n_classes, dtype=np.float64)

    def add_batch(self, y_pred, y_true):
        y_pred = np.asarray(y_pred).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)
        valid = y_true != self.ignore_index
        valid &= (y_true >= 0) & (y_true < self.label_mapping.size)
        valid &= (y_pred >= 0) & (y_pred < self.label_mapping.size)
        if not np.any(valid):
            return

        y_true = self.label_mapping[y_true[valid].astype(np.int64)]
        y_pred = self.label_mapping[y_pred[valid].astype(np.int64)]
        if self.empty_index is not None:
            occupied = y_true != self.empty_index
            y_true = y_true[occupied]
            y_pred = y_pred[occupied]
        for class_id in range(self.n_classes):
            true_class = y_true == class_id
            pred_class = y_pred == class_id
            self.tps[class_id] += np.logical_and(true_class, pred_class).sum()
            self.fps[class_id] += np.logical_and(~true_class, pred_class).sum()
            self.fns[class_id] += np.logical_and(true_class, ~pred_class).sum()
            self.gts[class_id] += true_class.sum()

    def get_stats(self):
        iou = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        acc = self.tps / (self.gts + 1e-5)
        valid_classes = self.gts > 0
        if self.empty_index is not None:
            valid_classes[self.empty_index] = False
        if np.any(valid_classes):
            miou = float(np.mean(iou[valid_classes]))
            macc = float(np.mean(acc[valid_classes]))
        else:
            miou = 0.0
            macc = 0.0
        return {
            'iou': iou,
            'acc': acc,
            'miou': miou,
            'macc': macc,
            'oa': float(np.sum(self.tps)) / max(float(np.sum(self.gts)), 1e-5),
            'valid_classes': valid_classes,
        }


class BinaryThresholdMetrics:
    """Binary voxel metrics evaluated at several probability thresholds."""

    def __init__(self, thresholds, positive_class, ignore_index=255):
        self.thresholds = np.asarray(thresholds, dtype=np.float64)
        if self.thresholds.ndim != 1 or self.thresholds.size == 0:
            raise ValueError('thresholds must be a non-empty one-dimensional sequence')
        self.n_classes = self.thresholds.size
        self.positive_class = int(positive_class)
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.n_classes, dtype=np.float64)
        self.fps = np.zeros(self.n_classes, dtype=np.float64)
        self.fns = np.zeros(self.n_classes, dtype=np.float64)
        self.gts = np.zeros(self.n_classes, dtype=np.float64)

    def add_batch(self, probabilities, targets):
        probabilities = np.asarray(probabilities).reshape(-1)
        targets = np.asarray(targets).reshape(-1)
        valid = targets != self.ignore_index
        probabilities = probabilities[valid]
        positives = targets[valid] == self.positive_class
        positive_count = positives.sum()
        for index, threshold in enumerate(self.thresholds):
            predicted = probabilities >= threshold
            self.tps[index] += np.logical_and(predicted, positives).sum()
            self.fps[index] += np.logical_and(predicted, ~positives).sum()
            self.fns[index] += np.logical_and(~predicted, positives).sum()
            self.gts[index] += positive_count

    def get_stats(self):
        precision = self.tps / (self.tps + self.fps + 1e-5)
        recall = self.tps / (self.tps + self.fns + 1e-5)
        iou = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        best_index = int(np.argmax(iou))
        return {
            'thresholds': self.thresholds,
            'precision': precision,
            'recall': recall,
            'iou': iou,
            'best_index': best_index,
        }


class PixelSegMetrics:
    """Semantic segmentation metrics on image pixels.

    Pixels with ignore_index are ignored. By default class 0 is also excluded
    from the mean, matching the common setup where 0 is unlabeled/background.
    """

    def __init__(self, n_classes, empty_index=0, ignore_index=255):
        self.n_classes = int(n_classes)
        self.empty_index = int(empty_index)
        self.ignore_index = int(ignore_index)
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.n_classes, dtype=np.float64)
        self.fps = np.zeros(self.n_classes, dtype=np.float64)
        self.fns = np.zeros(self.n_classes, dtype=np.float64)
        self.gts = np.zeros(self.n_classes, dtype=np.float64)

    def add_batch(self, y_pred, y_true):
        y_pred = np.asarray(y_pred).reshape(-1)
        y_true = np.asarray(y_true).reshape(-1)

        valid = (y_true != self.ignore_index) & (y_true != self.empty_index)
        valid &= (y_true >= 0) & (y_true < self.n_classes)
        y_pred = y_pred[valid]
        y_true = y_true[valid]
        if y_true.size == 0:
            return

        pred_valid = (y_pred >= 0) & (y_pred < self.n_classes)
        y_pred = np.where(pred_valid, y_pred, -1)
        for class_id in range(self.n_classes):
            if class_id == self.empty_index:
                continue
            true_class = y_true == class_id
            pred_class = y_pred == class_id
            tp = np.logical_and(true_class, pred_class).sum()
            fp = np.logical_and(~true_class, pred_class).sum()
            fn = np.logical_and(true_class, ~pred_class).sum()
            self.tps[class_id] += tp
            self.fps[class_id] += fp
            self.fns[class_id] += fn
            self.gts[class_id] += true_class.sum()

    def get_stats(self):
        iou = self.tps / (self.tps + self.fps + self.fns + 1e-5)
        acc = self.tps / (self.gts + 1e-5)
        valid_classes = np.ones(self.n_classes, dtype=bool)
        valid_classes[self.empty_index] = False
        valid_classes &= self.gts > 0
        if np.any(valid_classes):
            miou = float(np.mean(iou[valid_classes]))
            macc = float(np.mean(acc[valid_classes]))
        else:
            miou = 0.0
            macc = 0.0
        return {
            'iou': iou,
            'acc': acc,
            'miou': miou,
            'macc': macc,
            'valid_classes': valid_classes,
        }


class BEVSegMetrics(PixelSegMetrics):
    """Semantic segmentation metrics after projecting 3D occupancy to BEV."""

    def __init__(
        self,
        n_classes,
        empty_index=0,
        ignore_index=255,
        projection='mode',
        clean=True,
    ):
        super().__init__(n_classes, empty_index=empty_index, ignore_index=ignore_index)
        if projection not in ('mode', 'top'):
            raise ValueError(f'Unsupported BEV projection: {projection}')
        self.projection = projection
        self.clean = bool(clean)

    def add_occ_batch(self, y_pred_occ, y_true_occ):
        y_pred_bev = self.project_occ_batch(y_pred_occ)
        y_true_bev = self.project_occ_batch(y_true_occ)
        self.add_batch(y_pred_bev, y_true_bev)

    def project_occ_batch(self, occ):
        occ = np.asarray(occ)
        if occ.ndim == 3:
            occ = occ[None]
        if occ.ndim != 4:
            raise ValueError(f'Expected occupancy with shape [B, X, Y, Z], got {occ.shape}')

        bev = []
        for volume in occ:
            if self.projection == 'top':
                labels = self._project_top(volume)
            else:
                labels = self._project_mode(volume)
            if self.clean:
                labels = self._fill_empty_with_nearest(labels)
            bev.append(labels)
        return np.stack(bev, axis=0)

    def _valid_voxel_mask(self, volume):
        valid = volume != self.ignore_index
        valid &= volume != self.empty_index
        valid &= volume >= 0
        valid &= volume < self.n_classes
        return valid

    def _project_top(self, volume):
        valid = self._valid_voxel_mask(volume)
        labels = np.full(volume.shape[:2], self.empty_index, dtype=np.int64)
        has_label = valid.any(axis=-1)
        if not has_label.any():
            return labels

        idx_from_top = np.argmax(valid[..., ::-1], axis=-1)
        z_idx = volume.shape[-1] - 1 - idx_from_top
        x_idx = np.arange(volume.shape[0])[:, None]
        y_idx = np.arange(volume.shape[1])[None, :]
        labels[has_label] = volume[x_idx, y_idx, z_idx][has_label].astype(np.int64)
        return labels

    def _project_mode(self, volume):
        labels = np.full(volume.shape[:2], self.empty_index, dtype=np.int64)
        best_count = np.zeros(volume.shape[:2], dtype=np.int32)

        for class_id in range(self.n_classes):
            if class_id == self.empty_index:
                continue
            count = np.sum(volume == class_id, axis=-1)
            update = count > best_count
            labels[update] = class_id
            best_count[update] = count[update]
        return labels

    def _fill_empty_with_nearest(self, labels):
        labels = labels.copy()
        valid = labels != self.empty_index
        valid &= labels != self.ignore_index
        valid &= labels >= 0
        valid &= labels < self.n_classes
        if valid.all() or not valid.any():
            return labels

        if ndimage is None:
            return labels

        _, indices = ndimage.distance_transform_edt(~valid, return_indices=True)
        labels[~valid] = labels[tuple(indices[:, ~valid])]
        return labels


class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = np.zeros(self.n_classes)
        self.fps = np.zeros(self.n_classes)
        self.fns = np.zeros(self.n_classes)

        self.hist_ssc = np.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = np.zeros(self.n_classes, dtype=np.float32)
        self.cnt_class = np.zeros(self.n_classes, dtype=np.float32)
    
    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)
        
        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum
    
    def get_stats(self):
        if self.completion_tp != 0:
            precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            iou = self.completion_tp / (self.completion_tp + self.completion_fp + self.completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        
        iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)

        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }
        
    def get_score_completion(self, predict, target, nonempty=None):
        predict = np.copy(predict)
        target = np.copy(target)

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, n)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, n)
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = np.zeros(predict.shape)
        b_true = np.zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :] # ground truth
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]
            
            tp = np.array(np.where(np.logical_and(y_true == 1, y_pred == 1))).size
            fp = np.array(np.where(np.logical_and(y_true != 1, y_pred == 1))).size
            fn = np.array(np.where(np.logical_and(y_true == 1, y_pred != 1))).size

            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        return tp_sum, fp_sum, fn_sum
    
    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = np.copy(target)
        predict = np.copy(predict)
        _bs = predict.shape[0]
        _C = self.n_classes

        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, n)
        predict = predict.reshape(_bs, -1)  # (_bs, n)

        cnt_class = np.zeros(_C, dtype=np.int32)  # count for each class
        iou_sum = np.zeros(_C, dtype=np.float32)  # sum of iou for each class
        tp_sum = np.zeros(_C, dtype=np.int32)  # tp
        fp_sum = np.zeros(_C, dtype=np.int32)  # fp
        fn_sum = np.zeros(_C, dtype=np.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]
            y_pred = predict[idx, :]

            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
                y_true = y_true[np.where(np.logical_and(nonempty_idx == 1, y_true != 255))]
            
            for j in range(_C):
                tp = np.array(np.where(np.logical_and(y_true == j, y_pred == j))).size
                fp = np.array(np.where(np.logical_and(y_true != j, y_pred == j))).size
                fn = np.array(np.where(np.logical_and(y_true == j, y_pred != j))).size

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn
        
        return tp_sum, fp_sum, fn_sum
