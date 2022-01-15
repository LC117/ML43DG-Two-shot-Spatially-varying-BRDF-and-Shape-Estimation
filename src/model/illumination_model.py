import torch 
import numpy as np
import pytorch_lightning as pl

import torch.nn.functional as F
from torch import nn
from math import log2

import src.utils.rendering_layer as rl
import src.utils.sg_utils as sg 
from src.data.dataloader_lightning import TwoShotBrdfDataLightning
from src.utils.common_layers import INReLU


MAX_VAL = 2

"""
Notes:
    -> maybe interesting: pytorch_lightning "configure_shared_models"
"""

class IlluminationNetwork(pl.LightningModule):
    """ 
    Pytorch Implementation of: 
    https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py
    """
    def __init__(self, imgSize: int = 256, base_nf: int = 16, num_sgs: int = 24):
        super().__init__()
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.num_sgs = num_sgs # spherical gaussian
        self.axis_sharpness = torch.tensor(sg.setup_axis_sharpness(num_sgs), dtype=torch.float32)
        
        # env_net:
        # Define model:
        layers_needed = int(log2(256) - 2) # 256 = cam1.shape[1].value

        # enc: 
        enc_conv2d_list = []
        in_channels=11
        for i in range(layers_needed):
            out_channels = min(self.base_nf * (2 ** i), 256)
            enc_conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    #padding='same' TODO
                )
            )
            enc_conv2d_list.append(INReLU())
            # Set in_channels for the next layer:
            in_channels = out_channels
            
        self.enc_conv2d_list = torch.nn.ModuleList(enc_conv2d_list)

        # env_map:
        outputSize = self.num_sgs * 3
        self.env_map = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=256,
                kernel_size=3,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3, 
                stride=2
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=512,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(
                in_features=256,
                out_features=outputSize
            ),
            nn.Sigmoid(),          
        )
        
        # predictions:
        # reshape will be performed in forward
        return


    def forward(self, x):
        cam1, cam2, mask, normal, depth = x
        x = torch.cat([cam1, cam2, mask[:, :, :, 0:1], normal, depth], dim=-1) # shape: (None, 256, 256, 11) = (None, 256, 256, 3+3+1+3+1)
        
        # Feed trough Network ():
        for layer in self.enc_conv2d_list:
            x = layer(x)
        x = self.env_map(x)
        # Reshape result
        # predictions:
        sgs = torch.view(x * MAX_VAL, [-1, self.num_sgs, 3])
        
        # sgs_prep:
        # done in training step -> maybe here?
        
        return sgs # sphericalGaussainsShape
    
    def configure_optimizers(self):
        # half learning rate after half of the epochs: 
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        model_params = list(self.enc_conv2d_list.parameters()) + list(self.env_map.parameters())
        optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs // 2, gamma=0.5)
        # return [optimizer], [scheduler]
        return scheduler
    
    def general_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)
        
        # sgs_prep:
        batch_size = batch.shape[0]
        axis_sharpness = torch.tile(self.axis_sharpness[None, ...], torch.stack([batch_size, 1, 1]))
        sgs_joined = torch.cat([out, axis_sharpness], dim=-1)
        
        # if self.training:
        sgs_gt = torch.clip(targets, min=0.0, max=MAX_VAL)
        sgs_gt_joined = torch.cat([sgs_gt, axis_sharpness], dim=-1)
        
        # loss:
            # sgs: 
        sgs_loss = torch.mean(torch.nn.MSELoss(sgs_joined, sgs_gt_joined))
        
        return sgs_loss
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x for x in outputs]).mean()
        return avg_loss
    
    def training_step(self, batch, batch_idx):

        sgs_loss = self.general_step(batch, batch_idx)

        # batch_size = batch.shape[0]
        # viz:
        # TODO: Not sure if this is relevant for training, or only for logging examples to tensorboard..
        # renderer = rl.RenderingLayer(60, 0.7, torch.Size([batch_size, 256, 256, 3]))
        # sg_output = torch.zeros([batch_size, 256, 512, 3])
        # renderer.visualize_sgs(sgs_joined, sg_output)

        # if self.training:
        # sg_gt_output = torch.zeros_like(sg_output)
        # renderer.visualize_sgs(sgs_gt_joined, sg_gt_output, "sgs_gt")
                
        return {"loss": sgs_loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs)
        self.log('train_loss', avg_loss)
        return {'train_loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs)
        self.log('val_loss', avg_loss)
        return {'val_loss': avg_loss}
    
    
if __name__ == "__main__":
    # Training:
    overfit = 1
    
    model = IlluminationNetwork()
    
    trainer = pl.Trainer(
        weights_summary="full",
        profiler=True,
        max_epochs=1,
        progress_bar_refresh_rate=25, # to prevent notebook crashes in Google Colab environments,
        # gpus=1, # Use GPU if available
        overfit_batches=overfit
    )
    
    data = TwoShotBrdfDataLightning("illumination")
    trainer.fit(model, datamodule=data)
    
    