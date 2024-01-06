from typing import Optional
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .hnerv import HNeRV
import math
from torch.optim.lr_scheduler import _LRScheduler
from .hnerv_utils import loss_fn
import torch
import numpy as np
import torch.nn.functional as F

class CosineScheduler(_LRScheduler):
    def __init__(self,optimizer,up_ratio, up_pow, min_lr, last_epoch=-1) -> None:
        self.initial_lrs = [group['lr'] for group in optimizer.param_groups]
        self.up_ratio = up_ratio
        self.up_pow = up_pow
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.cur_epoch is not None:
            if self.cur_epoch < self.up_ratio:
                lr_mult = self.min_lr + (1. - self.min_lr) * (self.cur_epoch / self.up_ratio)** self.up_pow
            else:
                lr_mult = 0.5 * (math.cos(math.pi * (self.cur_epoch - self.up_ratio)/ (1 - self.up_ratio)) + 1.0)
        else:
            lr_mult = 1.
        new_lrs = []
        for lr in self.initial_lrs:
            if self.cur_epoch is None or self.cur_epoch<self.up_ratio:
                new_lrs.append(lr*lr_mult)
            else:
                new_lrs.append(lr*(1-self.min_lr)*lr_mult+lr * self.min_lr)
        return new_lrs

    def step(self, epoch = None, cur_ratio=None):
        self.cur_epoch = cur_ratio
        super().step(epoch)

'''
since for hnerv, the model architecture is determined by the dara length,
the args here has to be initialized by the dataset in advance
'''
class PLHNERV(pl.LightningModule):
    def __init__(self,args) -> None:
        super(PLHNERV, self).__init__()
        if 'pe' in args.embed or 'le' in args.embed:
            embed_param = 0
            embed_dim = int(args.embed.split('_')[-1]) * 2
            fc_param = np.prod([int(x) for x in args.fc_hw.split('_')])
        else:
            total_enc_strds = np.prod(args.enc_strds)
            embed_hw = args.final_size / total_enc_strds**2
            enc_dim1, embed_ratio = [float(x) for x in args.enc_dim.split('_')]
            embed_dim = int(embed_ratio * args.modelsize * 1e6 / args.full_data_length / embed_hw) if embed_ratio < 1 else int(embed_ratio) 
            embed_param = float(embed_dim) / total_enc_strds**2 * args.final_size * args.full_data_length
            
            enc_dim = f'{int(enc_dim1)}_{embed_dim}' 
            fc_param = (np.prod(args.enc_strds) // np.prod(args.dec_strds))**2 * 9
        decoder_size = args.modelsize * 1e6 - embed_param
        ch_reduce = 1. / args.reduce
        dec_ks1, dec_ks2 = [int(x) for x in args.ks.split('_')[1:]]
        fix_ch_stages = len(args.dec_strds) if args.saturate_stages == -1 else args.saturate_stages
        a =  ch_reduce * sum([ch_reduce**(2*i) * s**2 * min((2*i + dec_ks1), dec_ks2)**2 for i,s in enumerate(args.dec_strds[:fix_ch_stages])])
        b =  embed_dim * fc_param 
        c =  args.lower_width **2 * sum([s**2 * min(2*(fix_ch_stages + i) + dec_ks1, dec_ks2)  **2 for i, s in enumerate(args.dec_strds[fix_ch_stages:])])
        fc_dim = int(np.roots([a,b,c - decoder_size]).max())
        args.enc_dim = enc_dim
        args.fc_dim = fc_dim
        self.model = HNeRV(args)
        self.args = args
        self.automatic_optimization = False

    def forward(self, x, epoch):
        return self.model(x, epoch=epoch)
    
    def training_step(self, batch, batch_idx):
        img_data = batch['img']
        img_data, img_gt, inpaint_mask = self.model.transform_func(img_data)
        cur_input = batch['norm_idx'] if 'pe' in self.args.embed else img_data
        cur_epoch = self.current_epoch
        img_out, _, _ = self.forward(cur_input, cur_epoch)
        final_loss = loss_fn(img_out*inpaint_mask, img_gt*inpaint_mask, self.args.loss)   

        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        # batch_size is set to 1 for bunny and davis dataset, >1 for coco images
        cur_ratio = (cur_epoch+batch_idx/(1+len(self.trainer._data_connector._train_dataloader_source.dataloader())//self.args.batch_size))/self.args.epochs
            
        opt.zero_grad()
        self.manual_backward(final_loss)
        opt.step()  
        scheduler.step(cur_ratio=cur_ratio)  
 
        self.log('train_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)
        
        return final_loss
    
    def validation_step(self, batch, batch_idx):
        img_data = batch['img']
        img_data, img_gt, inpaint_mask = self.model.transform_func(img_data)
        cur_input = batch['norm_idx'] if 'pe' in self.args.embed else img_data
        cur_epoch = self.current_epoch
        img_out, _, _ = self.forward(cur_input, cur_epoch)
        psnr = self.psnr_fn_single(img_out.detach(), img_gt) 
        
        return {'psnr':psnr}
    
    def validation_epoch_end(self, outputs):
        psnr = torch.cat([x['psnr'] for x in outputs], 0).mean()
        self.log('val_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=self.args.enable_logger)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        up_ratio, up_pow, min_lr = [float(x) for x in self.args.lr_type.split('_')[1:]]
        scheduler = CosineScheduler(optimizer, up_ratio, up_pow, min_lr)
        return [optimizer], [scheduler]
    
    def psnr_fn_single(self,output, gt):
        l2_loss = F.mse_loss(output.detach(), gt.detach(),  reduction='none')
        psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
        return psnr.cpu()
    
    