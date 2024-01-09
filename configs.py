import argparse
from data.video.video_utils import add_data_specific_args
from models.hnerv.hnerv_utils import add_model_specific_args

parser = argparse.ArgumentParser(description='Implicit Neural Representation')
parser.add_argument('--project_name', type=str, default='INR', help='project name')
parser.add_argument('--debug', action='store_true', help='debug mode')
parser.add_argument('--outf', type=str, default='output', help='output folder')
parser.add_argument('--model', type=str, default='hnerv', help='model name')
parser.add_argument('--dataset', type=str, default='video', choices=['video', 'imagenet'],help='dataset name')
parser.add_argument('--gpus',type=int, default=1, help='number of gpus to use')

# parameters for training
parser.add_argument('-j', '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('-b', '--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--start_epoch', type=int, default=-1, help='starting epoch')
parser.add_argument('--not_resume', action='store_true', help='not resume from latest checkpoint')
parser.add_argument('-e', '--epochs', type=int, default=300, help='Epoch number')

parser.add_argument('--lr', type=float, default=0.001, help='learning rate, default=0.0002')
parser.add_argument('--lr_type', type=str, default='cosine_0.1_1_0.1', help='learning rate type, default=cosine')
parser.add_argument('--loss', type=str, default='L2', help='loss type, default=L2')
parser.add_argument('--out_bias', default='tanh', type=str, help='using sigmoid/tanh/0.5 for output prediction')  
parser.add_argument('--manualSeed', type=int, default=1, help='manual seed')
parser.add_argument('-d', '--distributed', action='store_true', default=False, help='distributed training,  added to suffix!!!!')

# parameters for logger
parser.add_argument('--enable_logger', action='store_true', default=False, help='enable logger')
parser.add_argument('--logger_type', type=str, default='csv_logger', choices=['csv_logger','wandb_logger','tensorboard_logger'], help='which logger to use')
parser.add_argument('--log_dir', type=str, default='logs', help='logging directory')

# parameters for evaluation


known_args, remaining_argv = parser.parse_known_args()
model = known_args.model
dataset = known_args.dataset

if dataset == 'video':
    from data.video.video_utils import add_data_specific_args  
elif dataset == 'imagenet':
    from data.imagenet.imagenet_utils import add_data_specific_args
if model == 'hnerv':
    from models.hnerv.hnerv_utils import add_model_specific_args

parser = add_data_specific_args(parser)
parser = add_model_specific_args(parser)
args = parser.parse_args()

import pprint
pprint.pprint(vars(args))
