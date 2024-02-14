import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import os
import numpy as np
import argparse
from copy import deepcopy
import torch.nn as nn

def loss_fn(pred, target, loss_type='L2', batch_average=True):
    target = target.detach()

    if loss_type == 'L2':
        loss = F.mse_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=False)
    elif loss_type == 'Fusion1':
        loss = 0.3 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion2':
        loss = 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.7 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion3':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion4':
        loss = 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion5':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion6':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion7':
        loss = 0.7 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion8':
        loss = 0.5 * F.mse_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.5 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1)
    elif loss_type == 'Fusion9':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion10':
        loss = 0.7 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.3 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion11':
        loss = 0.9 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.1 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    elif loss_type == 'Fusion12':
        loss = 0.8 * F.l1_loss(pred, target, reduction='none').flatten(1).mean(1) + 0.2 * (1 - ms_ssim(pred, target, data_range=1, size_average=False))
    return loss.mean() if batch_average else loss

def add_model_specific_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
    # parameters for encoder
    parser.add_argument('--embed', type=str, default='', help='empty string for HNeRV, and base value/embed_length for NeRV position encoding')
    parser.add_argument('--ks', type=str, default='0_1_5', help='kernel size for encoder and decoder')
    parser.add_argument('--enc_strds', type=int, nargs='+', default=[5,4,4,2,2], help='stride list for encoder')
    parser.add_argument('--enc_dim', type=str, default='64_16', help='enc latent dim and embedding ratio')
    parser.add_argument('--modelsize', type=float,  default=1.5, help='model parameters size: model size + embedding parameters')
    parser.add_argument('--saturate_stages', type=int, default=-1, help='saturate stages for model size computation')
    parser.add_argument('--block_params', type=str, default='1_1', help='residual blocks and percentile to save')

    # parameters for decoder
    parser.add_argument('--fc_hw', type=str, default='9_16', help='out size (h,w) for mlp')
    parser.add_argument('--reduce', type=float, default=1.2, help='chanel reduction for next stage')
    parser.add_argument('--lower_width', type=int, default=12, help='lowest channel width for output feature maps')
    parser.add_argument('--dec_strds', type=int, nargs='+', default=[5, 4, 4, 2, 2], help='strides list for decoder')
    parser.add_argument('--num_blks', type=str, default='1_1', help='block number for encoder and decoder')
    parser.add_argument("--conv_type", default=['convnext', 'pshuffel'], type=str, nargs="+",
        help='conv type for encoder/decoder', choices=['pshuffel', 'conv', 'convnext', 'interpolate'])
    parser.add_argument('--norm', default='none', type=str, help='norm layer for generator', choices=['none', 'bn', 'in', 'ln'])
    parser.add_argument('--act', type=str, default='gelu', help='activation to use', 
        choices=['relu', 'leaky', 'leaky01', 'relu6', 'gelu', 'swish', 'softplus', 'hardswish'])
    
    return parser
    
''''
Given a hnerv model, and the specifications about layers to be redefined, return a new model with the redefined layers
input params: model, specifications
output: model

params explanation:
    model: an HNeRV model
    specifications: a dictionary containing the weights data of decoders 1-5
'''
def redefine_hnerv(model, spec_dim, num, mode='tkd'):
    redefine_model = deepcopy(model)
    for dec_id, weight in spec_dim.items():
        tk_core, tk_factors, cp_factors = weight['tk_core'], weight['tk_factors'], weight['cp_factors']   
        bias_value = redefine_model.model.decoder[dec_id].conv.upconv._modules['0'].bias.data
        if mode == 'tkd':
            core_tensor_shape = tk_core[0].shape
            # factor_matrix_1_shape = tk_factors[0].shape
            factor_matrix_2_shape = tk_factors[1].shape
            factor_matrix_3_shape = tk_factors[2].shape
            redefine_model.model.decoder[dec_id].conv.upconv._modules['0']=nn.Sequential(
                nn.Conv2d(factor_matrix_3_shape[0], factor_matrix_3_shape[1], kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(core_tensor_shape[0], core_tensor_shape[1], kernel_size=core_tensor_shape[2], stride=1, padding=core_tensor_shape[2]//2, bias=False),
                nn.Conv2d(factor_matrix_2_shape[1], factor_matrix_2_shape[0], kernel_size=1, stride=1, padding=0, bias=True)
            )
            # load values into the model
            redefine_model.model.decoder[dec_id].conv.upconv._modules['0'][0].weight.data = tk_factors[2].t().unsqueeze(-1).unsqueeze(-1)
            redefine_model.model.decoder[dec_id].conv.upconv._modules['0'][1].weight.data = tk_core* tk_factors[0][num]
            redefine_model.model.decoder[dec_id].conv.upconv._modules['0'][2].weight.data = tk_factors[1].unsqueeze(-1).unsqueeze(-1)
            redefine_model.model.decoder[dec_id].conv.upconv._modules['0'][2].bias.data = bias_value
        elif mode == 'tkd-cpd':
            core_tensor_shape = tk_core[0].shape
            # factor_matrix_1_shape = tk_factors[0].shape
            factor_matrix_2_shape = tk_factors[1].shape
            factor_matrix_3_shape = tk_factors[2].shape
            cp_factors_1_shape = cp_factors[0].shape
            cp_factors_2_shape = cp_factors[1].shape
            cp_factors_3_shape = cp_factors[2].shape
            raise NotImplementedError
        else:
            raise NotImplementedError
    return redefine_model