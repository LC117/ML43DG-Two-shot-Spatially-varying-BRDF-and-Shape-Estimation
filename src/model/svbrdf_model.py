import pytorch_lightning as pl
import torch 
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import ReLU
from torch.nn.modules.activation import Sigmoid
from torch.nn import L1Loss
import util.sg_utils as sg 


from torch import nn
from math import log2
from util.common_layers import INReLU

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
        camera_pos = np.asarray([0, 0, 0]),
        light_pos = np.asarray([0, 0, 0]),
        light_color = np.asarray([1, 1, 1]),
        light_intensity_lumen = 45,
        num_sgs = 24,
        no_rendering_loss: bool = False,
    ):
        super().__init__()

        self.base_nf = base_nf
        self.imgSize = imgSize
        self.fov = fov
        self.distance_to_zero = distance_to_zero

        self.camera_pos = torch.from_numpy(
            camera_pos.reshape([1, 3]), dtype=np.float32
        )
        self.light_pos = torch.from_numpy(
            light_pos.reshape([1, 3]), dtype=np.float32
        )
        intensity = light_intensity_lumen / (4.0 * np.pi)
        light_col = light_color * intensity
        self.light_col = torch.from_numpy(
            light_col.reshape([1, 3]), dtype=tf.float32
        )

        self.num_sgs = num_sgs
        self.no_rendering_loss = no_rendering_loss

        self.axis_sharpness = torch.from_numpy(
            sg.setup_axis_sharpness(self.num_sgs), dtype=np.float32
        )

        # Define model
        self.model = network_architecture()

    def network_architecture(self):
        layers_needed = int(log2(self.imgSize) - 2)
        model = {}

        out_channels = 11
        skip_dims = []
        for i in range(layers_needed):
            skip_dims.append(out_channels)
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** i), 512)
            model["enc.conv{i}"] = Conv2d(
                in_channels, 
                out_channels, 
                4,
                stride = 2
            )

        for i in range(layers_needed):
            inv_i = layers_needed - i
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** (inv_i - 1)), 512)
            model["dec.tconv{i}"] = ConvTranspose2d(
                in_channels, 
                out_channels, 
                4,
                stride = 2
            )
            model["dec.conv{i}"] = Conv2d(
                out_channels + skip_dims[inv_i - 1], 
                out_channels, 
                3
            )
        model["activation"] = ReLU()

        in_channels = out_channels
        out_channels = 7
        model["output.conv"] = Conv2d(
            in_channels,
            out_channels,
            5
        )
        model["output.activation"] = Sigmoid()

        return model

    def forward(self, x):
        cam1, cam2, mask, normal, depth = x
        x = torch.cat([cam1, cam2, mask[:, :, :, 0:1], normal, depth], dim=-1) # shape: (None, 256, 256, 11) = (None, 256, 256, 3+3+1+3+1)
        
        model = self.model
        n_layers = int(log2(self.imgSize) - 2)
        skips = []
        
        # Encoding
        for i in range(n_layers):
            skips.append(x.clone())
            x = model["enc.conv{i}"](x)
            x = model["activation"](x)
        
        # Deconding
        for i in range(n_layers):
            inv_i = layers_needed - i
            x = model["dec.tconv{i}"](x)
            x = model["activation"](x)

            x = torch.cat((x, skips[inv_i - 1]))

            x = model["dec.conv{i}"](x)
            x = model["activation"](x)
        
        x = model["output.conv"](x)
        x = model["output.activation"](x)

        return x # BRDF Predictions

    def configure_optimizers(self):
        # half learning rate after half of the epochs: 
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        model_params = list()
        n_layers = int(log2(self.imgSize) - 2)
        for i in range(n_layers):
            model_params = model_params + model["enc.conv{i}"].parameters()
        for i in range(n_layers):
            model_params = model_params + model["dec.tconv{i}"].parameters()
            model_params = model_params + model["dec.conv{i}"].parameters()
        model_params = model_params + model["output.conv"].parameters()
        optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma=0.5)
        # Replace upper line later with lower line when trainer is defined!
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs // 2, gamma=0.5)
        return [optimizer], [scheduler]

    def general_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)

        loss_function = L1Loss()
        loss = loss_function(out, target)

        # TODO Add rendering loss!
        return loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x for x in outputs]).mean()
        return avg_loss
    
    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {"loss": sgs_loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs)
        self.log("train_loss", avg_loss)
        return {"train_loss": avg_loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs)
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss}


if __name__ == "__main__":
    # Training
    model = SVBRDF_Network()
    # TODO Define Trainer