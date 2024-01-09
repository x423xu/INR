import argparse

def add_data_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    parser.add_argument('--data_path', type=str, default='/home/xxy/Documents/data/imagenet/ILSVRC', help='data path for vid')
    parser.add_argument('--vid', type=str, default='imagenet', help='dataset id',)
    parser.add_argument('--crop_list', type=str, default='224_224', help='image crop size',)
    parser.add_argument('--resize_list', type=str, default='640_640', help='image resize size',)
    parser.add_argument('--val_freq', type=int, default=1, help='evaluation mode')
    
    return parser