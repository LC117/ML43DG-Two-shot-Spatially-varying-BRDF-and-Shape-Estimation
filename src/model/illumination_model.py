import pytorch_lightning as pl
import torch 
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

"""[summary]

Returns:
    [type]: [description]
    
Notes:
    -> check pytorch_lightning "configure_shared_models"
"""

class IlluminationNetwork(pl.LightningModule):
    """ Pytorch Implementation of: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py

    Args:
        pl ([type]): [description]
    """
    def __init__(self, imgSize: int = 256, base_nf: int = 16, num_sgs: int = 24):
        super().__init__()
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.num_sgs = num_sgs # spherical gaussian
        self.axis_sharpness = torch.tensor(sg.setup_axis_sharpness(num_sgs), dtype=np.float32)
        
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
        return


    def forward(self, x):
        cam1, cam2, mask, normal, depth = x
        x = torch.cat([cam1, cam2, mask[:, :, :, 0:1], normal, depth], dim=-1) # shape: (None, 256, 256, 11) = (None, 256, 256, 3+3+1+3+1)
        
        # Feed trough Network:
        for layer in self.enc_conv2d_list:
            x = layer(x)
        x = self.env_map(x)
        
        # Reshape result
        x = torch.view(x * MAX_VAL, [-1, self.num_sgs, 3])
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
        
        batch_size = batch.shape[0]
        
        axis_sharpness = torch.tile(self.axis_sharpness[None, ...], torch.stack([batch_size, 1, 1]))
        # axis_sharpness = tf.tile(tf.expand_dims(self.axis_sharpness, 0), tf.stack([batch_size, 1, 1]))  # add batch dim
        
        sgs_joined = torch.cat([out, axis_sharpness], dim=-1)    
        #     sgs_joined = tf.concat([sgs, axis_sharpness], -1, name="sgs")
        
        sgs_targets = torch.clip(targets, min=0.0, max=MAX_VAL)
        #     if self.training:
        #         sgs_gt = tf.clip_by_value(sgs_gt, 0.0, MAX_VAL)
        
        sgs_targets_joined = torch.cat([sgs_targets, axis_sharpness], dim=-1)
        #         sgs_gt_joined = tf.concat([sgs_gt, axis_sharpness], -1, name="sgs_gt")
        
        sgs_loss = torch.mean(torch.nn.MSELoss(sgs_joined, sgs_targets_joined))
        
        # with tf.variable_scope("loss"):
        #     with tf.variable_scope("sgs"):
        #         print("SGS Loss shapes (gt, pred):", sgs_gt.shape, sgs.shape)
        #         sgs_loss = tf.reduce_mean(l2_loss(sgs_gt, sgs), name="sgs_loss")
        
        
        #         print("sgs_loss", sgs_loss.shape)
        #         add_moving_summary(sgs_loss)
        #         tf.losses.add_loss(sgs_loss, tf.GraphKeys.LOSSES)

        # with tf.variable_scope("viz"):
        #     renderer = rl.RenderingLayer(60, 0.7, tf.TensorShape([None, 256, 256, 3]))
        #     sg_output = tf.zeros([batch_size, 256, 512, 3])
        #     renderer.visualize_sgs(sgs_joined, sg_output)

        #     if self.training:
        #         sg_gt_output = tf.zeros_like(sg_output)
        #         renderer.visualize_sgs(sgs_gt_joined, sg_gt_output, "sgs_gt")

        # print(sgs_loss)
        # self.cost = tf.losses.get_total_loss(name="total_costs")
        # print(self.cost)

        # add_moving_summary(self.cost)
        # add_param_summary((".*/W", ["histogram"]))  # monitor W

        # return self.cost
        
        
        


        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)
        
        return {"loss": loss}

    def validation_step(self, *args, **kwargs):
        return super().validation_step(*args, **kwargs)
    
    
    def test_step():
        pass
    