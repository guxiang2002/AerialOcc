import pytorch_lightning as pl
import numpy as np
import torch
from mmdet.datasets import build_dataset
from torch.utils.data.dataloader import DataLoader


def flexible_collate(batch):
    if len(batch) == 0:
        return batch

    elem = batch[0]

    if torch.is_tensor(elem):
        return torch.stack(batch, dim=0)

    if isinstance(elem, np.ndarray):
        return batch

    if isinstance(elem, (float, int, str)):
        return batch

    if isinstance(elem, dict):
        return {key: flexible_collate([sample[key] for sample in batch]) for key in elem}

    if isinstance(elem, tuple):
        transposed = zip(*batch)
        return tuple(flexible_collate(list(samples)) for samples in transposed)

    if isinstance(elem, list):
        if len(elem) == 0:
            return batch
        transposed = zip(*batch)
        return [flexible_collate(list(samples)) for samples in transposed]

    return batch


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config      
    ):
        super().__init__()
        self.trainset_config = config.data.train
        self.testset_config = config.data.test
        self.valset_config = config.data.val

        self.train_dataloader_config = config.train_dataloader_config
        self.test_dataloader_config = config.test_dataloader_config
        self.val_dataloader_config = config.test_dataloader_config
        self.config = config
        self.use_custom_collate = config.get('use_custom_collate', False)
    
    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = build_dataset(self.trainset_config)
            self.val_dataset = build_dataset(self.valset_config)
        if stage in (None, 'validate') and not hasattr(self, 'val_dataset'):
            self.val_dataset = build_dataset(self.valset_config)
        if stage in (None, 'test', 'predict'):
            self.test_dataset = build_dataset(self.testset_config)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_dataloader_config.batch_size,
            drop_last=True,
            num_workers=self.train_dataloader_config.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=flexible_collate if self.use_custom_collate else None)
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.val_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=flexible_collate if self.use_custom_collate else None)
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_dataloader_config.batch_size,
            drop_last=False,
            num_workers=self.test_dataloader_config.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=flexible_collate if self.use_custom_collate else None)
