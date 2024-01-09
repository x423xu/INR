import os
import pytorch_lightning as pl
from models.hnerv.hnerv_pl import PLHNERV
from configs import args
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
if args.dataset == 'imagenet':
    from data.imagenet.imagenet_pl import ImagenetDataModule as DataModule
elif args.dataset == 'video':
    from data.video.video_pl import VideoDataModule as DataModule

def init_version_name(args):
    import pytz
    from datetime import datetime
    toronto_tz = pytz.timezone('America/Toronto')
    utc_now = datetime.utcnow()
    toronto_now = utc_now.astimezone(toronto_tz)
    timestr = toronto_now.strftime("%Y%m%d-%H%M%S")

    version_name = f''
    version_name += f'{args.modelsize}'
    version_name += f'-{args.vid}'
    slr = "{:.0e}".format(args.lr)
    version_name += f'-lr{slr}'
    version_name += f'-bs{args.batch_size}'
    version_name += f'_{timestr}'
    return version_name

def init_logger(args, version_name):
    if args.enable_logger:
        if args.logger_type == 'csv_logger':
            logger = pl_loggers.CSVLogger(save_dir = args.log_dir, name=args.project_name, version=version_name)
        elif args.logger_type == 'wandb_logger':
            logger = pl_loggers.WandbLogger(project=args.project_name, save_dir=args.log_dir, version=version_name, entity='xxy')
        elif args.logger_type == 'tensorboard_logger':
            logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name=args.project_name, version=version_name)
    else:
        logger = None
    return logger

def init_checkpoint(args, version_name):
    dirpath = os.path.join(args.log_dir,version_name, 'checkpoints')
    ckpt_callback = ModelCheckpoint(dirpath=dirpath, 
                           save_last=True, 
                           save_top_k=1, 
                           monitor="val_psnr", 
                           mode="max", 
                           every_n_epochs=1, 
                           verbose=False)
    return ckpt_callback

def get_trainer_args(args):
    kwargs = {'max_epochs':args.epochs}
    if args.dataset=='video':
        kwargs['check_val_every_n_epoch'] = args.val_freq
    elif args.dataset=='imagenet':
        kwargs['check_val_every_n_epoch'] = args.val_freq
    if args.distributed:
        kwargs['accelerator'] = 'gpu'
        kwargs['devices'] = 2
        kwargs['strategy'] = 'ddp'
    else:
        kwargs['accelerator'] = 'gpu'
        kwargs['devices'] = args.gpus
    return kwargs


def main():
    pl.seed_everything(args.manualSeed)
    # init module
    # checkpoint_callback = ModelCheckpoint(dirpath='hdr_xyz',save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100)
    
    #!!! the model has to be initialized after the data module, since the model architecture is determined by the data length
    dm = DataModule(args)
    model = PLHNERV(args)
    print(model)
    version_name = init_version_name(args)
    logger = init_logger(args, version_name)
    ckpt_callback = init_checkpoint(args, version_name)
    # most basic trainer, uses good defaults
    kwargs = get_trainer_args(args)
    trainer = pl.Trainer(logger=logger,callbacks=[ckpt_callback], **kwargs)
    trainer.fit(model, dm)

main()