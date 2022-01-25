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
        device="cuda:0"
    ):
        super().__init__()

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
        self.model = self.network_architecture()

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

    def _masked_loss(self, loss, mask):
        return torch.where(torch.less_equal(mask, 1e-5), torch.zeros_like(loss), loss)

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
            padding_ = self._get_same_padding(256 / (2 ** i), 256 / (2 ** i), (4, 4), (2, 2))
            model[f"enc.conv{i}"] = Conv2d(
                in_channels, 
                out_channels, 
                4,
                padding = padding_[0],
                stride = 2
            )
            model[f"enc.conv.act{i}"] = INReLU(out_channels)

        for i in range(layers_needed):
            inv_i = layers_needed - i
            in_channels = out_channels
            out_channels = min(self.base_nf * (2 ** (inv_i - 1)), 512)
            print("dec.tconv", i, ": ", in_channels, " -> ", out_channels)
            padding_ = self._get_same_padding(256 / (2 ** i), 256 / (2 ** (inv_i - 1)), (4, 4), (2, 2))
            model[f"dec.tconv{i}"] = ConvTranspose2d(
                in_channels, 
                out_channels, 
                4,
                padding = padding_[0],
                stride = 2
            )
            model[f"dec.tconv.act{i}"] = INReLU(out_channels)
            print("-> dec.conv", i, ": ", out_channels + skip_dims[inv_i - 1], " -> ", out_channels)
            model[f"dec.conv{i}"] = Conv2d(
                out_channels + skip_dims[inv_i - 1], 
                out_channels, 
                3,
                padding = "same"
            )
            model[f"dec.conv.act{i}"] = INReLU(out_channels)
        print("======================================")
        in_channels = out_channels
        out_channels = 7
        model[f"output.conv"] = Conv2d(
            in_channels,
            out_channels,
            5,
            padding = "same"
        )
        model[f"output.activation"] = Sigmoid()

        return model

    def forward(self, x):
        cam1, cam2, mask, normal, depth = x
        x = torch.cat([cam1, cam2, mask, normal, depth], dim=1)
        
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
            x = torch.cat((x, skips[inv_i - 1]), dim=1)
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
        cam1, cam2, mask, normal, depth, sgs = batch["cam1"], batch["cam2"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"]
        x = cam1, cam2, mask, normal, depth
        gt_diffuse = batch["diffuse"]
        gt_specular = batch["specular"]
        gt_roughness = batch["roughness"]

        # Perform a forward pass on the network with inputs
        out = self.forward(x)
        pred_diffuse = out[:, 0:3]
        pred_specular = out[:, 3:6]
        pred_roughness = torch.unsqueeze(out[:, 6], 1)

        loss_function = L1Loss()

        loss_diffuse = loss_function(pred_diffuse, gt_diffuse)
        loss_specular = loss_function(pred_specular, gt_specular)
        loss_roughness = loss_function(pred_roughness, gt_roughness)

        loss = (loss_diffuse + loss_specular + loss_roughness) / 3.0
        return loss

        # Rendering Loss
        mask = mask[:, :, :, None]
        repeat = [1 for _ in range(len(mask.shape))]
        repeat[-1] = 3
        mask3 = torch.tile(mask, repeat)
        batch_size = cam1.shape[0]

        # Reshape because TF expected N x W x H x C
        # render()-function still takes TF-format
        diffuse_ = torch.moveaxis(pred_diffuse, 1, 3)
        specular_ = torch.moveaxis(pred_specular, 1, 3)
        roughness_ = torch.moveaxis(pred_roughness, 1, 3)
        depth_ = torch.unsqueeze(depth, 3)

        rendered = self.render(diffuse_, specular_, roughness_, normal, depth_, sgs, mask3)

        rerendered_log = torch.clip(torch.log(1.0 + torch.relu(rendered)), 0.0, 13.0)
        rerendered_log = torch.nan_to_num(rerendered_log)
        loss_log = torch.clip(torch.log(1.0 + torch.relu(cam1)), 0.0, 13.0)
        loss_log = torch.nan_to_num(loss_log)
        l1_err = loss_function(loss_log, rerendered_log)
        rerendered_loss = torch.mean(self._masked_loss(l1_err, mask3))

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
        self.log("train_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, "val_loss")
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss}

    def render(self, diffuse, specular, roughness, normal, depth, sgs, mask3):
        sdiff = torch.moveaxis(apply_mask(diffuse, mask3, "safe_diffuse", undefined=0.5), 3, 1)
        sspec = torch.moveaxis(apply_mask(specular, mask3, "safe_specular", undefined=0.04), 3, 1)
        srogh = torch.moveaxis(apply_mask(roughness, mask3[:, :, :, 0:1], "safe_roughness", undefined=0.4), 3, 1)
        snormal = torch.moveaxis(torch.where(
            torch.less_equal(mask3, 1e-5),
            torch.ones_like(normal) * torch.tensor([0.5, 0.5, 1.0]),
            normal
        ), 3, 1)
        batch_size = diffuse.shape[0]
        axis_sharpness = torch.tile(
            torch.unsqueeze(self.axis_sharpness, 0), (batch_size, 1, 1)
        )
        sgs_joined = torch.moveaxis(torch.cat((sgs, axis_sharpness), -1), 2, 1)
        renderer = RenderingLayer(
            self.fov,
            self.distance_to_zero,
            torch.Size((-1, 3, self.imgSize, self.imgSize)),
        )
        mask3 = torch.moveaxis(mask3, 3, 1)
        depth = torch.moveaxis(depth, 3, 1)

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
        rerendered = torch.moveaxis(torch.nan_to_num(rerendered), 1, 3)
        return rerendered


if __name__ == "__main__":
    # Training
    network = SVBRDF_Network()
    device = "cuda:0"

    trainer = pl.Trainer(
        weights_summary="full",
        max_epochs=100,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        gpus=1 if torch.cuda.is_available() else 0, # Use GPU if available
        profiler="simple"
    )

    data = TwoShotBrdfDataLightning(mode="svbrdf", overfit=True, num_workers=0)
    trainer.fit(network, train_dataloaders=data)

    test_sample = data.train_dataloader().dataset[0]
    cam1 = torch.unsqueeze(torch.tensor(test_sample["cam1"]), 0, device=device)
    cam2 = torch.unsqueeze(torch.tensor(test_sample["cam2"]), 0, device=device)
    mask = torch.unsqueeze(torch.tensor(test_sample["mask"]), 0, device=device)
    normal = torch.unsqueeze(torch.tensor(test_sample["normal"]), 0, device=device)
    depth = torch.unsqueeze(torch.tensor(test_sample["depth"]), 0, device=device)
    x = cam1, cam2, mask, normal, depth

    out = network.forward(x)
    diffuse = torch.squeeze(torch.moveaxis(out[:, 0:3], 1, 3)).cpu().detach().numpy()
    specular = torch.squeeze(torch.moveaxis(out[:, 3:6], 1, 3)).cpu().detach().numpy()
    roughness = torch.squeeze(out[:, 6]).cpu().detach().numpy()
    if not os.path.exists("Test_Results"):
        os.makedirs("Test_Results")

    gt_diffuse = test_sample["diffuse"]
    gt_specular = test_sample["specular"]
    gt_roughness = test_sample["roughness"]

    # save the diffuse map as rgb using matplotlib
    plt.imsave("Test_Results/diffuse.png", diffuse)
    # save the specular map as rgb using matplotlib
    plt.imsave("Test_Results/specular.png", specular)
    # save the roughness map using matplotlib
    plt.imsave("Test_Results/roughness.png", roughness, cmap="gray")

    # save ground truth
    plt.imsave("Test_Results/diffuse_gt.png", gt_diffuse)
    plt.imsave("Test_Results/specular_gt.png", gt_specular)
    plt.imsave("Test_Results/roughness_gt.png", gt_roughness, cmap="gray")