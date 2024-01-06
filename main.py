import pytorch_lightning as pl
from models.hnerv.hnerv_pl import PLHNERV
from data.video.video_pl import VideoDataModule
from configs import args
import pytorch_lightning.loggers as pl_loggers

def init_logger(args):
    if args.enable_logger:
        if args.logger_type == 'csv_logger':
            logger = pl_loggers.CSVLogger(save_dir = args.log_dir, name=args.project_name)
        elif args.logger_type == 'wandb_logger':
            logger = pl_loggers.WandbLogger(name = args.project_name, save_dir=args.log_dir)
        elif args.logger_type == 'tensorboard_logger':
            logger = pl_loggers.TensorBoardLogger(save_dir=args.log_dir, name=args.project_name)
    else:
        logger = None
    return logger

def init_checkpoint(args):
    pass

def main():
    # init module
    # checkpoint_callback = ModelCheckpoint(dirpath='hdr_xyz',save_last=True, save_top_k=1, monitor="train_psnr", mode="max", every_n_epochs=100)
    dm = VideoDataModule(args)
    model = PLHNERV(args)
    print(model)
    logger = init_logger(args)
    # most basic trainer, uses good defaults
    trainer = pl.Trainer(gpus=1, max_epochs=args.epochs, logger=logger)
    trainer.fit(model, dm)

main()