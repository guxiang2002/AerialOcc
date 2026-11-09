import os
import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from LightningTools.callbacks import TrainingTimeMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class StaticGraphDDPStrategy(DDPStrategy):
    def configure_ddp(self) -> None:
        super().configure_ddp()
        if hasattr(self.model, '_set_static_graph'):
            self.model._set_static_graph()
            print('DDP static graph enabled')


def get_next_numeric_tb_version(tb_root):
    if not os.path.isdir(tb_root):
        return 0

    versions = []
    for name in os.listdir(tb_root):
        if not name.startswith('version_'):
            continue
        suffix = name[len('version_'):]
        if suffix.isdigit():
            versions.append(int(suffix))

    if not versions:
        return 0
    return max(versions) + 1


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/aerialocc20k.py')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--load_from', default=None)
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default=None)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=None)
    parser.add_argument('--pretrain', action='store_true')

    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)

    args_dict = vars(args)
    load_from = args_dict.pop('load_from')
    cfg.update({key: value for key, value in args_dict.items() if value is not None})
    if 'log_folder' not in cfg:
        cfg['log_folder'] = 'aerialocc20k'
    if load_from is not None:
        cfg['load_from'] = load_from
    return args, cfg

if __name__ == '__main__':
    args, config = parse_config()
    print("[AerialOcc] " + ("inference/evaluation" if config.eval else "training"))
    output_root = config.get('output_root', 'logs')
    log_folder = os.path.join(output_root, config['log_folder'])
    misc.check_path(log_folder)

    tb_root = os.path.join(log_folder, 'tensorboard')
    misc.check_path(tb_root)
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard',
        version=get_next_numeric_tb_version(tb_root),
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    seed = config.seed
    pl.seed_everything(seed)
    num_gpu = torch.cuda.device_count()
    find_unused_parameters = config.get('find_unused_parameters', False)
    ddp_strategy_cls = StaticGraphDDPStrategy if config.get('ddp_static_graph', False) else DDPStrategy
    trainer_device_kwargs = dict(
        accelerator='gpu' if num_gpu > 0 else 'cpu',
        devices=[i for i in range(num_gpu)] if num_gpu > 1 else (1 if num_gpu == 1 else None),
    )
    if config.get('precision') is not None:
        trainer_device_kwargs['precision'] = config['precision']
    if num_gpu > 1:
        trainer_device_kwargs.update(
            strategy=ddp_strategy_cls(
                find_unused_parameters=find_unused_parameters
            ),
        )
    if config.eval or config.get('ckpt_path'):
        from aerialocc.training.checkpoint import disable_external_initialization
        disable_external_initialization(config.model)
        config.load_from = None
    model = pl_model(config)
    
    data_dm = DataModule(config)

    checkpoint_monitor = 'val/loss' if config.pretrain else 'val/mIoU'
    checkpoint_mode = 'min' if config.pretrain else 'max'
    callbacks = []
    checkpoint_dirpath = config.get('checkpoint_dirpath', None)
    if checkpoint_dirpath is not None:
        misc.check_path(checkpoint_dirpath)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dirpath,
        monitor=checkpoint_monitor,
        mode=checkpoint_mode,
        save_last=True,
        filename='best')
    callbacks.append(checkpoint_callback)

    checkpoint_every_n_train_steps = config.get(
        'checkpoint_every_n_train_steps', None)
    if checkpoint_every_n_train_steps:
        checkpoint_save_top_k = config.get('checkpoint_save_top_k', -1)
        callbacks.append(
            ModelCheckpoint(
                dirpath=checkpoint_dirpath,
                save_top_k=checkpoint_save_top_k,
                every_n_train_steps=checkpoint_every_n_train_steps,
                save_on_train_epoch_end=False,
                filename='step-{step}',
            )
        )
    
    if not config.eval:
        train_limit_kwargs = {}
        if config.get('max_epochs') is not None:
            train_limit_kwargs['max_epochs'] = int(config['max_epochs'])
        trainer = pl.Trainer(
            **train_limit_kwargs,
            **trainer_device_kwargs,
            max_steps=config.training_steps,
            gradient_clip_val=config.get('gradient_clip_val', 0.0),
            callbacks=callbacks + [
                LearningRateMonitor(logging_interval='step'),
                TrainingTimeMonitor(log_every_n_steps=config['log_every_n_steps'])
            ],
            logger=tb_logger,
            profiler=profiler,
            sync_batchnorm=num_gpu > 1,
            log_every_n_steps=config['log_every_n_steps'],
            check_val_every_n_epoch=config.get('check_val_every_n_epoch', 1),
            val_check_interval=config.get('val_check_interval', None),
        )
        trainer.fit(
            model=model,
            datamodule=data_dm,
            ckpt_path=config.get('ckpt_path', None),
        )
    else:
        eval_trainer_kwargs = {}
        if config.get('limit_test_batches', None) is not None:
            eval_trainer_kwargs['limit_test_batches'] = config['limit_test_batches']
        trainer = pl.Trainer(
            **trainer_device_kwargs,
            logger=tb_logger,
            profiler=profiler,
            **eval_trainer_kwargs,
        )
        trainer.test(
            model=model,
            datamodule=data_dm,
            ckpt_path=config.get('ckpt_path', None),
        )
