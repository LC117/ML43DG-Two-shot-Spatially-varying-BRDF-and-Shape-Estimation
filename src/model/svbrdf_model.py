from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from math import log2
from math import ceil
import numpy as np

import pytorch_lightning as pl
import torch 
import torch.nn.functional as F
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.activation import Sigmoid
from torch.nn import L1Loss
from torch import nn

from matplotlib import pyplot as plt

from src.data.dataloader_lightning import TwoShotBrdfDataLightning
from src.utils.common_layers import INReLU
import src.utils.sg_utils as sg
from src.utils.rendering_layer import *
from src.utils.common_layers import *
from src.utils.losses import masked_loss


class SVBRDF_Network(pl.LightningModule):
    """
    Pytorch Implementation of: 
    https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/brdf_network.py
    """
    def __init__(
        self,
        imgSize: int = 256,
        base_nf : int = 32,
        fov : int = 60,
        distance_to_zero : float = 0.7,
        camera_pos = np.array([0, 0, 0]),
        light_pos = np.array([0, 0, 0]),
        light_color = np.array([1, 1, 1]),
        light_intensity_lumen : int = 45,
        num_sgs : int = 24,
        no_rendering_loss : bool = False,
        device = "cuda:0"
    ):
        super().__init__()

        self.device__ = device
        
        self.base_nf = base_nf
        self.imgSize = imgSize
        self.fov = fov
        self.distance_to_zero = distance_to_zero

        self.camera_pos = torch.tensor(camera_pos.reshape([1, 3]), device=device)
        self.light_pos = torch.tensor(light_pos.reshape([1, 3]), device=device)
        intensity = light_intensity_lumen / (4.0 * np.pi)
        light_col = light_color * intensity
        self.light_col = torch.tensor(light_col.reshape([1, 3]), device=device)

        self.num_sgs = num_sgs
        self.no_rendering_loss = no_rendering_loss

        self.axis_sharpness = torch.tensor(sg.setup_axis_sharpness(self.num_sgs), device=device)

        # Define model
        layers_needed = int(log2(self.imgSize) - 2)
        self.enc_conv2d_steps = nn.ModuleList()
        self.dec_tconv2d_steps = nn.ModuleList()
        self.dec_conv2d_steps = nn.ModuleList()

        out_channels = 11
        skip_dims = []
        
        # Encoder:
        for i in range(layers_needed):
            skip_dims.append(out_channels)
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** i), 512)
            #print("enc.conv", i, ": ", in_channels, " -> ", out_channels)
            # padding_ = self._get_same_padding(256 / (2 ** i), 256 / (2 ** i), (4, 4), (2, 2))
            
            self.enc_conv2d_steps.append(nn.Sequential(
                Conv2d(
                    in_channels, 
                    out_channels, 
                    4,
                    padding = 1,
                    stride = 2
                ),
                INReLU(out_channels)
            ))
        
        # Decoder:
        for i in range(layers_needed):
            inv_i = layers_needed - i
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** (inv_i - 1)), 512)
            # print("dec.tconv", i, ": ", in_channels, " -> ", out_channels)
            # padding_ = self._get_same_padding(256 / (2 ** i), 256 / (2 ** (inv_i - 1)), (4, 4), (2, 2))
            
            self.dec_tconv2d_steps.append(nn.Sequential(
                ConvTranspose2d(
                    in_channels, 
                    out_channels, 
                    4,
                    padding = 1,
                    stride = 2
                ),
                INReLU(out_channels)
            ))

            #print("-> dec.conv", i, ": ", out_channels + skip_dims[inv_i - 1], " -> ", out_channels)
            self.dec_conv2d_steps.append(nn.Sequential(
                Conv2d(
                    out_channels + skip_dims[inv_i - 1], 
                    out_channels, 
                    3,
                    padding = "same"
                ),
                INReLU(out_channels)
            ))

        in_channels = out_channels
        out_channels = 7
        self.output_conv2d_step = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                5,
                padding = "same"
            ),
            Sigmoid()
        )


    def _get_same_padding(self, in_height, in_width, kernel_size, strides):
        out_height = ceil(float(in_height) / float(strides[0]))
        out_width  = ceil(float(in_width) / float(strides[1]))
        pad_along_height = max((out_height - 1) * strides[0] + kernel_size[0] - in_height, 0)
        pad_along_width = max((out_width - 1) * strides[1] + kernel_size[1] - in_width, 0)
        pad_top = pad_along_height // 2
        pad_bottom = pad_along_height - pad_top
        pad_left = pad_along_width // 2
        pad_right = pad_along_width - pad_left
        return (int(pad_top), int(pad_bottom), int(pad_left), int(pad_right))

    def _plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as 
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    def forward(self, x):
        cam1, cam2, mask, normal, depth = x
        x = torch.cat([cam1, cam2, normal, depth, mask], dim=1)

        n_layers = int(log2(self.imgSize) - 2)
        skips = []

        # Encoding
        for i in range(n_layers):
            skips.append(x)
            x = self.enc_conv2d_steps[i](x)
        
        # Decoding
        for i in range(n_layers):
            inv_i = n_layers - i
            x = self.dec_tconv2d_steps[i](x)
            x = torch.cat((x, skips[inv_i - 1]), dim = 1)
            x = self.dec_conv2d_steps[i](x)
        
        # Output
        x = self.output_conv2d_step(x)
        
        # brdf_prediction:
        diffuse = apply_mask(
            torch.clamp(x[:, 0:3, :, :], 0.0, 1.0),
            mask
        )

        specular = apply_mask(
            torch.clamp(x[:, 3:6, :, :], 40 / 255, 1.0),
            mask
        )

        roughness = apply_mask(
            torch.clamp(x[:, 6:7, :, :], 0.004, 1.0),
            mask
        )

        return diffuse, specular, roughness

    def configure_optimizers(self):
        # half learning rate after half of the epochs: 
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        model_params = []
        for step_ in self.enc_conv2d_steps:
            model_params = model_params + list(step_.parameters())
        for step_ in self.dec_tconv2d_steps:
            model_params = model_params + list(step_.parameters())
        for step_ in self.dec_conv2d_steps:
            model_params = model_params + list(step_.parameters())
        model_params = model_params + list(self.output_conv2d_step.parameters())
        optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        step_size = self.trainer.max_epochs // 2
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size if step_size > 0 else 1,
            gamma=0.5)
        return [optimizer], [scheduler]
        
    def general_step(self, batch, batch_idx):
        cam1, cam2, mask, normal, depth, sgs = batch["cam1"], batch["cam2"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"]
        x = cam1, cam2, mask, normal, depth
        gt_diffuse = batch["diffuse"]
        gt_specular = batch["specular"]
        gt_roughness = batch["roughness"]

        # Perform a forward pass on the network with inputs
        pred_diffuse, pred_specular, pred_roughness = self.forward(x)
        # pred_diffuse = out[:, 0:3]
        # pred_specular = out[:, 3:6]
        # pred_roughness = torch.unsqueeze(out[:, 6], 1)

        loss_function = L1Loss()

        loss_diffuse = loss_function(pred_diffuse, gt_diffuse)
        loss_specular = loss_function(pred_specular, gt_specular)
        loss_roughness = loss_function(pred_roughness, gt_roughness)

        if self.no_rendering_loss:
            loss = (loss_diffuse + loss_specular + loss_roughness) / 3.0
            return loss
        
        # TODO With rendering loss!
        # mask = mask[:, 0:1]
        repeat = [1 for _ in range(len(mask.shape))]
        repeat[1] = 3
        mask3 = torch.tile(mask, repeat)
        
        # batch_size = cam1.shape[0]

        rendered = self.render(pred_diffuse, pred_specular, pred_roughness, normal, depth, sgs, mask3)

        rerendered_log_pred = torch.clip(torch.log(1.0 + torch.relu(rendered)), 0.0, 13.0)
        # rerendered_log = torch.nan_to_num(rerendered_log)
        loss_log_target = torch.clip(torch.log(1.0 + torch.relu(cam1)), 0.0, 13.0)
        # loss_log = torch.nan_to_num(loss_log)
        
        # l1_err = loss_function(loss_log, rerendered_log)
        rerendered_loss = masked_loss(rerendered_log_pred, loss_log_target, mask, loss_function)

        loss = (loss_diffuse + loss_specular + loss_roughness + rerendered_loss) / 4.0
        return loss
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        self.log("my_loss", loss, logger=True, on_step=True, on_epoch=True)
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, "loss")
        #self._plot_grad_flow(self.named_parameters())
        self.log("train_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, "val_loss")
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss}
    
    def render(self, diffuse, specular, roughness, normal, depth, sgs, mask3):
        sdiff = apply_mask(diffuse, mask3, "safe_diffuse", undefined=0.5)
        sspec = apply_mask(specular, mask3, "safe_specular", undefined=0.04)
        srogh = apply_mask(roughness, mask3[:, 0:1], "safe_roughness", undefined=0.4)

        # normal_scale_ = torch.moveaxis(torch.ones_like(normal), 1, 3)
        # normal_scale_ = normal_scale_ * torch.tensor([0.5, 0.5, 1.0], device=self.device__)
        # normal_scale_ = torch.moveaxis(normal_scale_, 3, 1)
        
        snormal = torch.where(
            torch.less_equal(mask3, 1e-5),
            torch.ones_like(normal) * torch.tensor([0.5, 0.5, 1.], device=self.device__)[None, :, None, None],
            normal
        )
        
        batch_size = diffuse.shape[0]
        
        axis_sharpness = torch.tile(
            torch.unsqueeze(self.axis_sharpness, 0), (batch_size, 1, 1)
        )
        
        sgs_joined = torch.cat((sgs, axis_sharpness), -1)
        
        renderer = RenderingLayer(
            self.fov,
            self.distance_to_zero,
            output_shape = torch.Size((batch_size, 3, self.imgSize, self.imgSize)),
            device = self.device__
        )
        
        # sgs_joined = torch.swapaxes(sgs_joined, 1, 2)
        rerendered = renderer.call(
            sdiff,
            sspec,
            srogh,
            snormal,
            depth,  # Depth is still in 0 - 1 range
            mask3[:, 0:1],
            self.camera_pos,
            self.light_pos,
            self.light_col,
            sgs_joined,
        )
        rerendered = apply_mask(rerendered, mask3, "masked_rerender")
        # rerendered = torch.nan_to_num(rerendered)
        assert not torch.any(rerendered == torch.nan)
        return rerendered


if __name__ == "__main__":
    print("================ SV-BRDF Network ================")
    # Training
    model = SVBRDF_Network(no_rendering_loss = False, device = "cuda:0" if torch.cuda.is_available() else "cpu")

    trainer = pl.Trainer(
        weights_summary="full",
        max_epochs=10,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        gpus= 1 if torch.cuda.is_available() else 0, # Use GPU if available
        profiler="simple",
    )

    data = TwoShotBrdfDataLightning(mode="svbrdf", overfit=True, num_workers=0, batch_size=3)
    trainer.fit(model, train_dataloaders=data)
