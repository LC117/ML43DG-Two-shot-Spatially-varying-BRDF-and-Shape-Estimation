from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from math import log2
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
from torchvision.transforms.functional import center_crop

from src.utils.common_layers import INReLU
from src.data.dataloader_lightning import TwoShotBrdfDataLightning
import src.utils.sg_utils as sg 

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

        self.camera_pos = torch.from_numpy(camera_pos.reshape([1, 3]))
        self.light_pos = torch.from_numpy(light_pos.reshape([1, 3]))
        intensity = light_intensity_lumen / (4.0 * np.pi)
        light_col = light_color * intensity
        self.light_col = torch.from_numpy(light_col.reshape([1, 3]))

        self.num_sgs = num_sgs
        self.no_rendering_loss = no_rendering_loss

        self.axis_sharpness = torch.from_numpy(sg.setup_axis_sharpness(self.num_sgs))

        # Define model
        self.model = self.network_architecture()

    def network_architecture(self):
        layers_needed = int(log2(self.imgSize) - 2)
        model = {}

        print("======================================")
        out_channels = 11
        skip_dims = []
        for i in range(layers_needed):
            skip_dims.append(out_channels)
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** i), 512)
            print("enc.conv", i, ": ", in_channels, " -> ", out_channels)
            model[f"enc.conv{i}"] = Conv2d(
                in_channels, 
                out_channels, 
                4,
                stride = 2
            )
            model[f"enc.conv.act{i}"] = INReLU(out_channels)

        for i in range(layers_needed):
            inv_i = layers_needed - i
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** (inv_i - 1)), 512)
            print("dec.tconv", i, ": ", in_channels, " -> ", out_channels)
            model[f"dec.tconv{i}"] = ConvTranspose2d(
                in_channels, 
                out_channels, 
                4,
                stride = 2
            )
            model[f"dec.tconv.act{i}"] = INReLU(out_channels)
            print("-> dec.conv", i, ": ", out_channels + skip_dims[inv_i - 1], " -> ", out_channels)
            model[f"dec.conv{i}"] = Conv2d(
                out_channels + skip_dims[inv_i - 1], 
                out_channels, 
                3
            )
            model[f"dec.conv.act{i}"] = INReLU(out_channels)
        print("======================================")
        in_channels = out_channels
        out_channels = 7
        model[f"output.conv"] = Conv2d(
            in_channels,
            out_channels,
            5
        )
        model[f"output.activation"] = Sigmoid()

        return model

    def forward(self, x):
        cam1, cam2, mask, normal, depth, sgs = x
        x = torch.cat([cam1, cam2, mask[:, :, :, None], normal, depth[:, :, :, None]], dim=-1)
        x = torch.swapaxes(x, 1, 3)
        
        model = self.model
        n_layers = int(log2(self.imgSize) - 2)
        skips = []
        
        # Encoding
        for i in range(n_layers):
            skips.append(x.clone())
            x = model[f"enc.conv{i}"](x)
            x = model[f"enc.conv.act{i}"](x)

        # Deconding
        for i in range(n_layers):
            inv_i = n_layers - i
            x = model[f"dec.tconv{i}"](x)
            x = model[f"dec.tconv.act{i}"](x)

            img_w, img_h = (x.shape[2], x.shape[3])
            skip_ = center_crop(skips[inv_i - 1], (img_w, img_h))
            x = torch.cat((x, skip_), dim=1)

            x = model[f"dec.conv{i}"](x)
            x = model[f"dec.conv.act{i}"](x)
        
        x = model[f"output.conv"](x)
        x = model[f"output.activation"](x)

        return x # BRDF Predictions

    def configure_optimizers(self):
        # half learning rate after half of the epochs: 
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        model = self.model
        model_params = list()
        n_layers = int(log2(self.imgSize) - 2)
        for i in range(n_layers):
            model_params = model_params + list(model[f"enc.conv{i}"].parameters())
            model_params = model_params + list(model[f"enc.conv.act{i}"].parameters())
        for i in range(n_layers):
            model_params = model_params + list(model[f"dec.tconv{i}"].parameters())
            model_params = model_params + list(model[f"dec.tconv.act{i}"].parameters())
            model_params = model_params + list(model[f"dec.conv{i}"].parameters())
            model_params = model_params + list(model[f"dec.conv.act{i}"].parameters())
        model_params = model_params + list(model[f"output.conv"].parameters())
        optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        step_size = self.trainer.max_epochs // 2
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size if step_size > 0 else 1,
            gamma=0.5)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.trainer.max_epochs // 2, gamma=0.5)
        return [optimizer], [scheduler]

    def general_step(self, batch, batch_idx):
        #images, targets = batch
        x = batch["cam1"], batch["cam2"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"]
        gt_diffuse = torch.swapaxes(batch["diffuse"], 1, 3)
        gt_specular = torch.swapaxes(batch["specular"], 1, 3)
        gt_roughness = torch.unsqueeze(batch["roughness"], 1)

        # Perform a forward pass on the network with inputs
        out = self.forward(x)

        loss_function = L1Loss()
        loss = loss_function(out, targets)

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
    network = SVBRDF_Network()

    trainer = pl.Trainer(
        weights_summary="full",
        max_epochs=100,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        # gpus=1, # Use GPU if available
    )

    data = TwoShotBrdfDataLightning(mode="all", overfit=True)
    trainer.fit(network, train_dataloaders=data)