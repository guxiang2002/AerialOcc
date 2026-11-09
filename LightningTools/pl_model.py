import os
import torch
import numpy as np
import pytorch_lightning as pl
import torch.distributed as dist
from .basemodel import LightningBaseModel
from .metric import (
    BEVSegMetrics,
    OccupiedVoxelSemMetrics,
    RemappedSemanticIoUMetrics,
    SSCMetrics,
)
from mmdet3d.models import build_model
from mmcv.runner.checkpoint import load_checkpoint


def load_model_init_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(checkpoint, dict) or 'state_dict' not in checkpoint:
        load_checkpoint(model, checkpoint_path, map_location='cpu')
        return

    source_state = checkpoint['state_dict']
    target_state = model.state_dict()
    load_state = {}
    skipped_shape = []
    skipped_missing = []

    for key, value in source_state.items():
        clean_key = key
        if clean_key.startswith('model.'):
            clean_key = clean_key[len('model.'):]
        if clean_key.startswith('module.'):
            clean_key = clean_key[len('module.'):]

        if clean_key not in target_state:
            skipped_missing.append(clean_key)
            continue
        if tuple(target_state[clean_key].shape) != tuple(value.shape):
            skipped_shape.append((clean_key, tuple(value.shape), tuple(target_state[clean_key].shape)))
            continue
        load_state[clean_key] = value

    missing, unexpected = model.load_state_dict(load_state, strict=False)
    print(
        'loaded model init checkpoint: '
        f'loaded={len(load_state)}, missing={len(missing)}, unexpected={len(unexpected)}, '
        f'skipped_shape={len(skipped_shape)}, skipped_missing={len(skipped_missing)}'
    )
    if skipped_shape:
        preview = ', '.join(
            f'{name}: {src}->{dst}' for name, src, dst in skipped_shape[:8])
        print(f'skipped shape mismatch preview: {preview}')


class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        if 'load_from' in config and config['load_from']:
            if os.path.exists(config['load_from']):
                load_model_init_checkpoint(self.model, config['load_from'])
            else:
                raise FileNotFoundError(config['load_from'])
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']

        self.train_metrics = SSCMetrics(config['num_class'])
        self.val_metrics = SSCMetrics(config['num_class'])
        self.test_metrics = SSCMetrics(config['num_class'])
        self.train_occ_sem_metrics = OccupiedVoxelSemMetrics(config['num_class'])
        self.val_occ_sem_metrics = OccupiedVoxelSemMetrics(config['num_class'])
        self.test_occ_sem_metrics = OccupiedVoxelSemMetrics(config['num_class'])
        mcop_mapping = config.get('mcop_label_mapping')
        self.mcop_class_names = config.get('mcop_class_names', [])
        self.test_mcop_metrics = (
            RemappedSemanticIoUMetrics(
                mcop_mapping,
                self.mcop_class_names,
                empty_index=0 if config.get('mcop_occupied_only', False) else None,
            )
            if mcop_mapping is not None else None
        )
        self.bev_seg_enable = config.get('bev_seg_enable', False)
        self.test_bev_seg_metrics = BEVSegMetrics(
            config['num_class'],
            projection=config.get('bev_seg_projection', 'mode'),
            clean=config.get('bev_seg_clean', True),
            empty_index=config.get('bev_seg_empty_index', 0),
            ignore_index=config.get('bev_seg_ignore_index', 255),
        )
        self.save_path = config.get('save_path')
        self.test_mapping = config.get('test_mapping', False)
        self.save_pred_as_raw_labels = config.get('save_pred_as_raw_labels', False)
        self.save_occ_prob_compressed = config.get(
            'save_occ_prob_compressed', True
        )
        self.save_output_verbose = config.get('save_output_verbose', True)
        compact_to_raw_label = config.get('compact_to_raw_label')
        self.compact_to_raw_label = None if compact_to_raw_label is None else np.asarray(compact_to_raw_label, dtype=np.int32)
        self.paper_class_indices = config.get('paper_class_indices', None)
        self.paper_class_names = config.get('paper_class_names', None)
        self.eval_name = config.get('eval_name', 'all')
        self.pretrain = config.get('pretrain', False)
        self.empty_cache_interval = int(config.get('empty_cache_interval', 0))
    
    def forward(self, data_dict):
        return self.model(data_dict)

    def get_report_miou(self, stats):
        if self.paper_class_indices is None:
            return float(stats["iou_ssc_mean"]), None

        paper_indices = np.asarray(self.paper_class_indices, dtype=np.int64)
        paper_ious = stats['iou_ssc'][paper_indices]
        return float(np.mean(paper_ious)), paper_ious

    @staticmethod
    def _distributed_is_initialized():
        return dist.is_available() and dist.is_initialized()

    def _get_distributed_metric_stats(self, metric):
        if not self._distributed_is_initialized():
            return metric.get_stats()

        counts = np.concatenate([
            np.asarray(
                [metric.completion_tp, metric.completion_fp, metric.completion_fn],
                dtype=np.float64,
            ),
            metric.tps.astype(np.float64),
            metric.fps.astype(np.float64),
            metric.fns.astype(np.float64),
        ])
        counts_tensor = torch.as_tensor(counts, dtype=torch.float64, device=self.device)
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        counts = counts_tensor.detach().cpu().numpy()

        completion_tp, completion_fp, completion_fn = counts[:3]
        offset = 3
        tps = counts[offset:offset + self.num_class]
        offset += self.num_class
        fps = counts[offset:offset + self.num_class]
        offset += self.num_class
        fns = counts[offset:offset + self.num_class]

        if completion_tp != 0:
            precision = completion_tp / (completion_tp + completion_fp)
            recall = completion_tp / (completion_tp + completion_fn)
            iou = completion_tp / (completion_tp + completion_fp + completion_fn)
        else:
            precision, recall, iou = 0, 0, 0
        iou_ssc = tps / (tps + fps + fns + 1e-5)
        return {
            "precision": precision,
            "recall": recall,
            "iou": iou,
            "iou_ssc": iou_ssc,
            "iou_ssc_mean": np.mean(iou_ssc[1:]),
        }

    def _get_distributed_seg_stats(self, metric):
        if not self._distributed_is_initialized():
            return metric.get_stats()

        counts = np.concatenate([
            metric.tps.astype(np.float64),
            metric.fps.astype(np.float64),
            metric.fns.astype(np.float64),
            metric.gts.astype(np.float64),
        ])
        counts_tensor = torch.as_tensor(counts, dtype=torch.float64, device=self.device)
        dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
        counts = counts_tensor.detach().cpu().numpy()

        n_classes = metric.tps.size
        offset = 0
        metric.tps = counts[offset:offset + n_classes]
        offset += n_classes
        metric.fps = counts[offset:offset + n_classes]
        offset += n_classes
        metric.fns = counts[offset:offset + n_classes]
        offset += n_classes
        metric.gts = counts[offset:offset + n_classes]
        return metric.get_stats()
    
    def training_step(self, batch, batch_idx):
        # Avoid Lightning recursively scanning NumPy images in img_metas.
        batch_size = int(batch['gt_occ'].shape[0])
        output_dict = self.forward(batch)
        loss_dict = output_dict['losses']
        loss = 0
        for key, value in loss_dict.items():
            self.log(
                "train/"+key,
                value.detach(),
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size)
            loss += value
            
        self.log("train/loss",
            loss.detach(),
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size)
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()
            
            self.train_metrics.add_batch(pred, gt_occ)
            self.train_occ_sem_metrics.add_batch(pred, gt_occ)

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        if (
            self.empty_cache_interval > 0
            and torch.cuda.is_available()
            and (self.global_step + 1) % self.empty_cache_interval == 0
        ):
            torch.cuda.empty_cache()
    
    def validation_step(self, batch, batch_idx):
        batch_size = int(batch['gt_occ'].shape[0])
        if self.pretrain:
            output_dict = self.model.forward_train(batch)
        else:
            output_dict = self.forward(batch)
        loss_dict = output_dict.get('losses')
        if loss_dict is not None:
            loss = 0
            for key, value in loss_dict.items():
                self.log(
                    "val/" + key,
                    value.detach(),
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
                loss += value

            self.log(
                "val/loss",
                loss.detach(),
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        
        if not self.pretrain:
            pred = output_dict['pred'].detach().cpu().numpy()
            gt_occ = output_dict['gt_occ'].detach().cpu().numpy()

            self.val_metrics.add_batch(pred, gt_occ)
            self.val_occ_sem_metrics.add_batch(pred, gt_occ)
    
    def validation_epoch_end(self, outputs):
        if self.pretrain:
            return
        if getattr(self.trainer, 'sanity_checking', False):
            self.val_metrics.reset()
            return

        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]
        # metric_list = [("val", self.val_metrics)]
        
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = self._get_distributed_metric_stats(metric)
            miou, _ = self.get_report_miou(stats)

            self.log("{}/mIoU".format(prefix), torch.tensor(miou, dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            if int(getattr(self.trainer, 'global_rank', 0)) == 0:
                print(
                    f'\n[{prefix} metrics] '
                    f'mIoU={miou:.6f} '
                    f'IoU={float(stats["iou"]):.6f} '
                    f'Precision={float(stats["precision"]):.6f} '
                    f'Recall={float(stats["recall"]):.6f}'
                )
            metric.reset()

        occ_sem_metric_list = [
            ("train", self.train_occ_sem_metrics),
            ("val", self.val_occ_sem_metrics),
        ]
        for prefix, metric in occ_sem_metric_list:
            stats = self._get_distributed_seg_stats(metric)
            self.log(
                f"{prefix}/occ_sem_mIoU",
                torch.tensor(stats["miou"], dtype=torch.float32),
                sync_dist=True,
            )
            self.log(
                f"{prefix}/occ_sem_mAcc",
                torch.tensor(stats["macc"], dtype=torch.float32),
                sync_dist=True,
            )
            if int(getattr(self.trainer, 'global_rank', 0)) == 0:
                print(
                    f'\n[{prefix} occ_sem metrics] '
                    f'mIoU={float(stats["miou"]):.6f} '
                    f'mAcc={float(stats["macc"]):.6f}'
                )
            metric.reset()
    
    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict['pred'].detach().cpu().numpy()
        pred_occ_prob = output_dict.get('pred_occ_prob')
        pred_occ_prob_class_id = output_dict.get('pred_occ_prob_class_id')
        pred_occ_prob_class_ids = output_dict.get('pred_occ_prob_class_ids')
        gt_occ = output_dict['gt_occ']
        if isinstance(gt_occ, (list, tuple)) and len(gt_occ) == 1 and gt_occ[0] is None:
            gt_occ = None
        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()
        else:
            gt_occ = None
            
        if self.save_path is not None:
            if pred_occ_prob is not None:
                pred_occ_prob = pred_occ_prob.detach().cpu().numpy().astype(np.float16)
            if self.save_pred_as_raw_labels and self.compact_to_raw_label is not None:
                output_voxels = self.compact_to_raw_label[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch['img_metas']['sequence'][0]
            frame_id = batch['img_metas']['frame_id'][0]
            altitude = batch['img_metas'].get('altitude')
            if altitude is not None:
                if isinstance(altitude, (list, tuple)):
                    altitude = altitude[0]
                sequence_id = os.path.join(sequence_id, str(altitude))
            save_folder = "{}/sequences/{}/predictions".format(self.save_path, sequence_id)
            save_file = os.path.join(save_folder, "{}.label".format(frame_id))
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                output_voxels.tofile(f)
                if self.save_output_verbose:
                    print('\n save to {}'.format(save_file))
            if pred_occ_prob is not None:
                prob_folder = "{}/sequences/{}/occ_probs".format(self.save_path, sequence_id)
                prob_file = os.path.join(prob_folder, "{}.npz".format(frame_id))
                os.makedirs(prob_folder, exist_ok=True)
                probability_payload = {
                    'prob': pred_occ_prob[0],
                    'class_id': np.int32(
                        -1
                        if pred_occ_prob_class_id is None
                        else pred_occ_prob_class_id
                    ),
                }
                if pred_occ_prob_class_ids is not None:
                    class_ids = np.asarray(
                        pred_occ_prob_class_ids, dtype=np.int32
                    )
                    probability_payload['class_ids'] = class_ids
                    if self.compact_to_raw_label is not None:
                        probability_payload['raw_class_ids'] = (
                            self.compact_to_raw_label[class_ids].astype(np.int32)
                        )
                save_probability = (
                    np.savez_compressed
                    if self.save_occ_prob_compressed
                    else np.savez
                )
                save_probability(prob_file, **probability_payload)
                if self.save_output_verbose:
                    print('\n save occ prob to {}'.format(prob_file))
            
        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)
            self.test_occ_sem_metrics.add_batch(pred, gt_occ)
            if self.test_mcop_metrics is not None:
                self.test_mcop_metrics.add_batch(pred, gt_occ)
            if self.bev_seg_enable:
                self.test_bev_seg_metrics.add_occ_batch(pred, gt_occ)
    
    def test_epoch_end(self, outputs):
        metric_list = [("test", self.test_metrics)]
        # metric_list = [("val", self.val_metrics)]
        metrics_list = metric_list
        for prefix, metric in metrics_list:
            stats = metric.get_stats()
            miou, paper_ious = self.get_report_miou(stats)

            for name, iou in zip(self.class_names, stats['iou_ssc']):
                print(name + ":", iou)

            self.log("{}/mIoU".format(prefix), torch.tensor(miou, dtype=torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"], dtype=torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"], dtype=torch.float32), sync_dist=True)
            metric.reset()

        seg_stats = self._get_distributed_seg_stats(self.test_occ_sem_metrics)
        self.log(
            "test/occ_sem_mIoU",
            torch.tensor(seg_stats["miou"], dtype=torch.float32),
            sync_dist=True,
        )
        self.log(
            "test/occ_sem_mAcc",
            torch.tensor(seg_stats["macc"], dtype=torch.float32),
            sync_dist=True,
        )
        self.log(
            "test/occ_sem_OA",
            torch.tensor(seg_stats["oa"], dtype=torch.float32),
            sync_dist=True,
        )
        print("occ_sem/mIoU:", float(seg_stats["miou"]))
        print("occ_sem/mAcc:", float(seg_stats["macc"]))
        print("occ_sem/OA:", float(seg_stats["oa"]))
        for name, iou, acc, valid in zip(
            self.class_names,
            seg_stats['iou'],
            seg_stats['acc'],
            seg_stats['valid_classes'],
        ):
            if not valid:
                continue
            print("occ_sem/{}_iou:".format(name), float(iou))
            print("occ_sem/{}_acc:".format(name), float(acc))
        self.test_occ_sem_metrics.reset()

        if self.test_mcop_metrics is not None:
            mcop_stats = self._get_distributed_seg_stats(self.test_mcop_metrics)
            for metric_name, value in (
                ('mIoU', mcop_stats['miou']),
                ('mAcc', mcop_stats['macc']),
                ('OA', mcop_stats['oa']),
            ):
                self.log(
                    'test/occ_mcop/{}'.format(metric_name),
                    torch.tensor(float(value), dtype=torch.float32),
                    sync_dist=True,
                )
                print('occ_mcop/{}:'.format(metric_name), float(value))
            for name, iou, acc, valid in zip(
                self.mcop_class_names,
                mcop_stats['iou'],
                mcop_stats['acc'],
                mcop_stats['valid_classes'],
            ):
                if not valid:
                    continue
                print('occ_mcop/{}_iou:'.format(name), float(iou))
                print('occ_mcop/{}_acc:'.format(name), float(acc))
            self.test_mcop_metrics.reset()

        if self.bev_seg_enable:
            bev_seg_stats = self._get_distributed_seg_stats(self.test_bev_seg_metrics)
            self.log(
                "test/seg/mIoU",
                torch.tensor(bev_seg_stats["miou"], dtype=torch.float32),
                sync_dist=True,
            )
            self.log(
                "test/seg/mAcc",
                torch.tensor(bev_seg_stats["macc"], dtype=torch.float32),
                sync_dist=True,
            )
            print("seg/mIoU:", float(bev_seg_stats["miou"]))
            print("seg/mAcc:", float(bev_seg_stats["macc"]))
            for name, iou, acc, valid in zip(
                self.class_names,
                bev_seg_stats['iou'],
                bev_seg_stats['acc'],
                bev_seg_stats['valid_classes'],
            ):
                if not valid:
                    continue
                print("seg/{}_iou:".format(name), float(iou))
                print("seg/{}_acc:".format(name), float(acc))
            self.test_bev_seg_metrics.reset()
