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
from PIL import Image
import wandb

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
            new_lrs.append(lr*lr_mult)
            # if self.cur_epoch is None or self.cur_epoch<self.up_ratio:
            #     new_lrs.append(lr*lr_mult)
            # else:
            #     new_lrs.append(lr*(1-self.min_lr)*lr_mult+lr * self.min_lr)
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
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        img_data = batch['img']
        import matplotlib.pyplot as plt

        # Assuming img_data is a tensor containing the image data
        # if self.args.debug:
        # first_img = img_data[0].cpu().numpy()  # Convert tensor to numpy array
        # first_img = np.transpose(first_img, (1, 2, 0))  # Transpose to (H, W, C)
        # first_img = (first_img - np.min(first_img)) / (np.max(first_img) - np.min(first_img))  # Normalize to 0-1
        # first_img = (first_img * 255).astype(np.uint8)  # Scale to 0-255
        # plt.imshow(first_img)
        # plt.show()

        img_data, img_gt, inpaint_mask = self.model.transform_func(img_data)
        cur_input = batch['norm_idx'] if 'pe' in self.args.embed else img_data
        cur_epoch = self.current_epoch 
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
    
        img_out, _, _ = self.forward(cur_input)
        final_loss = loss_fn(img_out*inpaint_mask, img_gt*inpaint_mask, self.args.loss)  

        # first_img = img_gt[0].detach().cpu().numpy()  # Convert tensor to numpy array
        # first_img = np.transpose(first_img, (1, 2, 0))  # Transpose to (H, W, C)
        # first_img = (first_img - np.min(first_img)) / (np.max(first_img) - np.min(first_img))  # Normalize to 0-1
        # first_img = (first_img * 255).astype(np.uint8)  # Scale to 0-255
        # plt.imshow(first_img)
        # plt.show() 

        opt.zero_grad()
        self.manual_backward(final_loss)    
        opt.step()   
        if 'cosine' in self.args.lr_type:  
            cur_ratio = (cur_epoch+batch_idx/len(self.trainer._data_connector._train_dataloader_source.dataloader()))/self.args.epochs
            scheduler.step(cur_ratio=cur_ratio) 
        if 'step' in self.args.lr_type:
            scheduler.step()
          
        
        train_psnr = self.psnr_fn_single(img_out.detach(), img_gt)
        if not hasattr(self, 'train_psnrs'):
            self.train_psnrs = []
        self.train_psnrs.append(train_psnr)

        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr, on_step=True, on_epoch=False, prog_bar=True, logger=self.args.enable_logger, sync_dist=True if self.args.distributed else False,rank_zero_only=True)
 
        self.log('train_loss', final_loss, on_step=True, on_epoch=True, prog_bar=True, logger=self.args.enable_logger if self.args.dataset == 'imagenet' else False, sync_dist=True if self.args.distributed else False,rank_zero_only=True)
        
        return final_loss
    
    def on_train_epoch_end(self, outputs=None):
        avg_psnr = torch.cat(self.train_psnrs).mean()
        self.log('avg_psnr', avg_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=self.args.enable_logger)
       
        self.train_psnrs = []
        if not hasattr(self, 'max_train_psnr'):
            self.max_train_psnr = 0
        if avg_psnr > self.max_train_psnr:
            self.max_train_psnr = avg_psnr
        self.log('max_train_psnr', self.max_train_psnr, on_step=False, on_epoch=True, prog_bar=True, logger=self.args.enable_logger,sync_dist=True if self.args.distributed else False,rank_zero_only=True)


    def validation_step(self, batch, batch_idx):
        img_data = batch['img']
        img_data, img_gt, inpaint_mask = self.model.transform_func(img_data)
        cur_input = batch['norm_idx'] if 'pe' in self.args.embed else img_data
        img_out, _, _ = self.forward(cur_input)
        psnr = self.psnr_fn_single(img_out.detach(), img_gt) 

        # let's log the first image to see what it looks like
        if self.trainer.is_global_zero:
            if batch_idx == 0:
                first_img = img_data[1].cpu().numpy()  # Convert tensor to numpy array
                first_img = np.transpose(first_img, (1, 2, 0))  # Transpose to (H, W, C)
                first_img = (first_img - np.min(first_img)) / (np.max(first_img) - np.min(first_img))  # Normalize to 0-1
                first_img = (first_img * 255).astype(np.uint8)  # Scale to 0-255

                first_out = img_out[1].detach().cpu().numpy()  # Convert tensor to numpy array
                first_out = np.transpose(first_out, (1, 2, 0))  # Transpose to (H, W, C)
                first_out = (first_out - np.min(first_out)) / (np.max(first_out) - np.min(first_out))
                first_out = (first_out * 255).astype(np.uint8)

                cat_img = np.concatenate((first_img, first_out), axis=1)
                cat_pil = Image.fromarray(cat_img)
                self.val_imgs=cat_pil

        return {'psnr':psnr}
    
    def validation_epoch_end(self, outputs):
        psnr = torch.cat([x['psnr'] for x in outputs], 0).mean()
        self.log('val_psnr', psnr, on_step=False, on_epoch=True, prog_bar=True, logger=self.args.enable_logger, sync_dist=True if self.args.distributed else False, rank_zero_only=True) 
        # lest's see what the output_image look like
        if self.trainer.is_global_zero and self.args.enable_logger and self.args.logger_type == 'wandb_logger':
            self.logger.experiment.log({"output_image": [wandb.Image(self.val_imgs, caption="input vs output")]})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        if 'cosine' in self.args.lr_type:
            up_ratio, up_pow, min_lr = [float(x) for x in self.args.lr_type.split('_')[1:]]
            scheduler = CosineScheduler(optimizer, up_ratio, up_pow, min_lr)
            scheduler.step(cur_ratio=0)
        if 'step' in self.args.lr_type:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)        
        return [optimizer], [scheduler]
    
    def psnr_fn_single(self,output, gt):
        l2_loss = F.mse_loss(output.detach(), gt.detach(),  reduction='none')
        psnr = -10 * torch.log10(l2_loss.flatten(start_dim=1).mean(1) + 1e-9)
        return psnr.cpu()
    
    