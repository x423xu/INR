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
class CatVideoDataSet(Dataset):
    def __init__(self, args, videos=['bunny', 'Beauty', 'Bosphorus','ShakeNDry']):
    # def __init__(self, args, videos=['bunny1', 'bunny2', 'bunny3','bunny4']):
        self.args = args
        video_dict = {}
        for v in videos:
            args.data_path = '/'.join(args.data_path.split('/')[:-1]) + '/' + v
            if os.path.isfile(args.data_path):
                video = decord.VideoReader(args.data_path)
            else:
                video = [os.path.join(args.data_path, x) for x in sorted(os.listdir(args.data_path))]
            video = sorted(video, key=get_number_suffix)
            video_dict[v] = video
        self.min_len = min([len(v) for v in video_dict.values()])
        video_dict = {k: v[:self.min_len] for k,v in video_dict.items()}
        self.video_dict = video_dict
        self.crop_list, self.resize_list = args.crop_list, args.resize_list
        # import pdb; pdb.set_trace; from IPython import embed; embed()     
        first_frame = self.img_transform(list(self.img_load(0).values())[0])
        self.final_size = first_frame.size(-2) * first_frame.size(-1)

    def img_load(self, idx):
        img_dict = {}
        for k,v in self.video_dict.items():
            if isinstance(v, list):
                img = read_image(v[idx])/255
            else:
                img = v[idx].permute(-1,0,1)/255
            img_dict[k] = img
        return img_dict

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
        # return len(self.video_dict[list(self.video_dict.keys())[0]])
        return self.min_len

    def __getitem__(self, idx):
        img_dict = self.img_load(idx)
        for k,v in img_dict.items():
            img_dict[k] = self.img_transform(v)
        norm_idx = float(idx) / self.min_len
        sample = {'img': img_dict, 'idx': idx, 'norm_idx': norm_idx} 
        return sample
