from torchvision.transforms.functional import center_crop, resize
from torchvision.io import read_image
from torch.nn.functional import interpolate
from torch.utils.data import Dataset
import os
import decord
decord.bridge.set_bridge('torch')
import re
import torch

def get_number_suffix(filename):
    match = re.search(r'_(\d+)\.png$', filename)
    return int(match.group(1)) if match else 0

'''
args:
    data_path: path to video
    crop_list: crop size (h, w) or -1
    resize_list: resize size (h, w) or -1
'''
class VideoDataSet(Dataset):
    def __init__(self, args):
        self.args = args
        if os.path.isfile(args.data_path):
            self.video = decord.VideoReader(args.data_path)
        else:
            self.video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]

        # Resize the input video and center crop
        self.video = sorted(self.video, key=get_number_suffix)
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        # import pdb; pdb.set_trace; from IPython import embed; embed()     
        first_frame = self.img_transform(self.img_load(0))
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        if isinstance(self.video, list):
            img = read_image(self.video[idx])
        else:
            img = self.video[idx].permute(-1,0,1)
        return img / 255.

    def img_transform(self, img):
        if self.crop_list != '-1': 
            crop_h, crop_w = [int(x) for x in self.crop_list.split('_')[:2]]
            if 'last' not in self.crop_list:
                img = center_crop(img, (crop_h, crop_w))
        if self.resize_list != '-1':
            if '_' in self.resize_list:
                resize_h, resize_w = [int(x) for x in self.resize_list.split('_')]
                img = interpolate(img, (resize_h, resize_w), 'bicubic')
            else:
                resize_hw = int(self.resize_list)
                img = resize(img, resize_hw,  'bicubic')
        if 'last' in self.crop_list:
            img = center_crop(img, (crop_h, crop_w))
        return img

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):
        tensor_image = self.img_transform(self.img_load(idx))
        norm_idx = float(idx) / len(self.video)
        sample = {'img': tensor_image, 'idx': idx, 'norm_idx': norm_idx}
        
        return sample