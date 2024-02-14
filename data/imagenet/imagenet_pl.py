'''
This file is used to load the imagenet dataset.
'''
import os
from typing import Any, Callable, Optional
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

class ImageNetDataSet(ImageFolder):
    def __init__(self, root, transform):
        super().__init__(root, transform=transform)

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        norm_index = index / self.__len__()
        path, target = self.samples[index]
        img_name = os.path.basename(path)
        img_name = os.path.splitext(img_name)[0]
        img = self.loader(path)
        # print(img.width, img.height)
        if self.transform is not None:
            img = self.transform(img)   
        sample = {'img': img, 'idx': index, 'norm_idx': norm_index, 'img_name':img_name}
        return sample

def worker_init_fn(worker_id):
    """
    Re-seed each worker process to preserve reproducibility
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

class ImagenetDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.crop_list = args.crop_list
        self.resize_list = args.resize_list
        transform = self._get_transform()

        # random select 10 as training set, 200 as validation set
        full_dataset = ImageNetDataSet(root = os.path.join(self.args.data_path, 'Data/CLS-LOC/train'), transform=transform)
        indices = list(range(len(full_dataset)))
        np.random.shuffle(indices)
        selected_train_indices = indices[:10000]
        self.train_dataset = Subset(full_dataset, selected_train_indices)
        selected_val_indices = indices[:200]
        self.val_dataset = Subset(full_dataset, selected_val_indices)
        
        first_frame = self.train_dataset[0]['img']
        final_size = first_frame.size(-2) * first_frame.size(-1)
        # here we keep the same size as the hnerv does (132).
        self.args.full_data_length = 132
        self.args.final_size = final_size
    
            
    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True, drop_last=False, worker_init_fn=worker_init_fn)
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(self.train_dataset, batch_size=1, shuffle=False)
        return test_loader
    
    def _get_transform(self):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
            else:
                raise NotImplementedError
        transform_list = []
        transform_list.append(transforms.ToTensor())
        if self.crop_list != '-1':
            if 'last' not in self.crop_list:
                transform_list.append(transforms.CenterCrop((crop_h, crop_w)))
                if self.resize_list != '-1':
                    transform_list.append(transforms.Resize((resize_h, resize_w)))
            else:
                if self.resize_list != '-1':
                    transform_list.append(transforms.Resize((resize_h, resize_w)))
                transform_list.append(transforms.CenterCrop((crop_h, crop_w)))      
        # transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        transform = transforms.Compose(transform_list)
        return transform