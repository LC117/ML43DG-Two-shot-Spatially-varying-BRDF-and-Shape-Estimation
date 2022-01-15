import pytorch_lightning as pl
import torch 
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.linear import Linear
import util.sg_utils as sg 


from torch import nn
from math import log2
from util.common_layers import INReLU

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
        self.axis_sharpness = torch.tensor(sg.setup_axis_sharpness(num_sgs), dtype=np.float32)
        
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
                    padding='same'
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
                out_channels=256
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
                in_channels=512,
                out_channels=256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(
                in_channels=256,
                out_channels=outputSize
            ),
            nn.Sigmoid()          
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
        
        return x
    
    def configure_optimizers(self):
        # half learning rate after half of the epochs: 
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        model_params = list(self.enc_conv2d_list.parameters()) + list(self.env_map.parameters())
        optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs // 2, gamma=0.5)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
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

        # viz:
        # TODO: Not sure if this is relevant for training, or only for logging examples to tensorboard..
        renderer = rl.RenderingLayer(60, 0.7, tf.TensorShape([None, 256, 256, 3]))
        sg_output = tf.zeros([batch_size, 256, 512, 3])
        renderer.visualize_sgs(sgs_joined, sg_output)

        # if self.training:
        sg_gt_output = torch.zeros_like(sg_output)
        renderer.visualize_sgs(sgs_gt_joined, sg_gt_output, "sgs_gt")

        # print(sgs_loss)
        # self.cost = tf.losses.get_total_loss(name="total_costs")
        # print(self.cost)

        # add_moving_summary(self.cost)
        # add_param_summary((".*/W", ["histogram"]))  # monitor W

        # return self.cost
        
        return {"loss": sgs_loss}

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    
    def test_step():
        pass
    