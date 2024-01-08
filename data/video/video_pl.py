import torch
import pytorch_lightning as pl
from .video_dataset import VideoDataSet
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from .video_utils import data_split

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

class VideoDataModule(pl.LightningDataModule):
    def __init__(self, args, preload=False):
        super().__init__()
        self.args = args
        self.preload = preload

        # to get the full data length and image size, the video data set inits here
        self.full_dataset = VideoDataSet(self.args, self.preload)
        final_size = self.full_dataset.final_size
        full_data_length = len(self.full_dataset)
        split_num_list = [int(x) for x in self.args.data_split.split('_')]
        self.train_ind_list, val_ind_list = data_split(list(range(full_data_length)), split_num_list, self.args.shuffle_data, 0)     
        self.args.full_data_length = full_data_length
        self.args.final_size = final_size
    
    def setup(self, stage=None):
        if stage=='fit':
            self.train_dataset = Subset(self.full_dataset, self.train_ind_list)
            self.val_dataset = self.full_dataset
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset) if self.args.distributed else None
    
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=(self.sampler is None),
            num_workers=self.args.workers, pin_memory=True, sampler=self.sampler, drop_last=False, worker_init_fn=worker_init_fn)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        return val_loader
    