import os

import torch
import numpy as np
import pytorch_lightning as pl

import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from math import log2

import src.utils.rendering_layer as rl
import src.utils.sg_utils as sg
from src.data.dataloader_lightning import TwoShotBrdfDataLightning
from src.utils.common_layers import INReLU, uncompressDepth, div_no_nan
from src.utils.merge_conv import MergeConv

MAX_VAL = 2

"""
Notes:
    -> maybe interesting: pytorch_lightning "configure_shared_models"
"""


class ShapeNetwork(pl.LightningModule):
    """
    Pytorch Implementation of:
    https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py
    """

    def __init__(self, imgSize: int = 256, base_nf: int = 32, downscale_steps: int = 4, consistency_loss: float = 0.5):
        super().__init__()
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.downscale_steps = downscale_steps
        self.consistency_loss_factor = consistency_loss
        self.enable_consistency = consistency_loss != 0

        #self.num_sgs = num_sgs  # spherical gaussian
        #self.axis_sharpness = torch.tensor(sg.setup_axis_sharpness(num_sgs), dtype=torch.float32, device=self.device)

        # env_net:
        # Define model:
        #layers_needed = int(log2(256) - 2)  # 256 = cam1.shape[1].value

        # encoder:
        encoder_layers = []
        inp_channels = 4
        for i in range(self.downscale_steps):
            out_channels = min(base_nf * (2 ** i), 256)
            encoder_layers.append(MergeConv(
                inp_channels,
                out_channels,
                4,
                2,
                True,
                INReLU,
            ))
            inp_channels = out_channels
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        for i in range(self.downscale_steps):
            inv_i = self.downscale_steps - i
            nf = min(base_nf * (2 ** (inv_i - 1)), 256)

            decoder_layers.append(MergeConv(
                inp_channels,
                nf,
                4,
                2,
                False,
                INReLU,
            ))
            inp_channels = nf
        self.decoder = nn.Sequential(*decoder_layers)

        self.geom_estimation = torch.nn.Sequential(
            torch.nn.Conv2d(inp_channels, 4, 5, padding='same'),
            torch.nn.Sigmoid()
        )
        return

    def forward(self, x):
        cam1, cam2, mask = x

        cam1_cf = torch.cat((cam1, mask), dim=1)
        cam2_cf = torch.cat((cam2, mask), dim=1)


        x = self.encoder((cam1_cf, cam2_cf, None))
        _, _, x = self.decoder(x)
        x = self.geom_estimation(x)

        normal = x[:, 0:3, :, :] * 2 - 1
        normal = normal / torch.norm(normal, dim=1, keepdim=True)
        normal = (normal * 0.5 + 0.5) * mask
        
        depth = x[:, 3:4, :, :] * mask  # + (1 - mask) * torch.ones_like(x)

        return normal, depth

    def masked_loss(self, pred, target, mask, loss):
        
        pred = pred * mask
        target = target * mask
        loss = loss(pred, target)
        
        return loss

    def configure_optimizers(self):
        # half learning rate after half of the epochs:
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        # model_params = list(self.enc_conv2d_block.parameters()) + list(self.env_map.parameters())
        #model_params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.geom_estimation.parameters())

        # optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0002, betas=(0.5, 0.999))
        step_size = self.trainer.max_epochs // 2
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size if step_size > 0 else 1,
            gamma=0.5)
        return [optimizer], [scheduler]

    def general_step(self, batch, batch_idx):
        # cam1, cam2, mask
        mask = batch["mask"]
        
        x = batch["cam1"], batch["cam2"], batch["mask"]
        # targets = batch["normal"], batch["depth"].reshape(-1, self.imgSize, self.imgSize, 1)
        normal_gt, depth_gt = batch["normal"], batch["depth"]#.unsqueeze(-1)

        # Perform a forward pass on the network with inputs
        normal, depth = self.forward(x)

        # In the paper the following loss is a L2 loss, but the tf implementation uses L1:
        normal_l1_loss = self.masked_loss(normal * 2 - 1, normal_gt * 2 - 1, mask, torch.nn.L1Loss())
        depth_l1_loss = self.masked_loss(depth, depth_gt, mask, torch.nn.L1Loss())
        consistency_loss = 0
        
        if self.enable_consistency:
            
            near = uncompressDepth(1)
            far = uncompressDepth(0)
            d = uncompressDepth(depth)
            depth_inv = div_no_nan(d - near, far - near)

            # calculate the gradient dx, dy and dz of the predicted depth map
            # declare the sobelfilters as torch.FloatTensor type
            sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device).detach()
            sobel_x = sobel_x.view((1, 1, 3, 3))
            sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device).detach()
            sobel_y = - sobel_y.view((1, 1, 3, 3))

            dx = F.conv2d(depth_inv, sobel_x, padding=1, stride=1)
            dy = F.conv2d(depth_inv, sobel_y, padding=1, stride=1)

            texel_size = 1 / self.imgSize
            # create a tensor of ones of shape and type of out[..., 0]
            dz = torch.ones_like(dx) * texel_size * 2

            cn = torch.cat([dx, dy, dz], dim=1)
            cn = cn / torch.norm(cn, dim=1, keepdim=True)
            cn = cn * 0.5 + 0.5
            
            # In the paper the following loss is a L1 loss, but the tf implementation uses L2:
            consistency_loss = self.consistency_loss_factor * self.masked_loss(cn, normal, mask, torch.nn.MSELoss())

        shape_loss = depth_l1_loss + normal_l1_loss + consistency_loss

        return shape_loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        sgs_loss = self.general_step(batch, batch_idx)
        self.log("my_loss", sgs_loss, logger=True, on_step=True, on_epoch=True)

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
        avg_loss = self.general_end(outputs, "loss")
        self.log("train_loss", avg_loss)

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, "val_loss")
        self.log("val_loss", avg_loss)
        return {"val_loss": avg_loss}

    def test_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = self.general_end(outputs, "test_loss")
        self.log("test_loss", avg_loss)
        return {"test_loss": avg_loss}


if __name__ == "__main__":
    # Training:

    model = ShapeNetwork(consistency_loss=0.5)#downscale_steps=2, base_nf=4)

    trainer = pl.Trainer(
        #weights_summary="full",
        max_epochs=30,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        gpus=0,  # Use GPU if available
        profiler="simple",
        #precision=16,
    )

    data = TwoShotBrdfDataLightning(mode="shape", overfit=True, num_workers=4, batch_size=8, persistent_workers=True, pin_memory=True)

    trainer.fit(model, train_dataloaders=data)

    test_sample = data.train_dataloader().dataset[1]
    print("test sample", test_sample.keys(), test_sample["mask"].shape)
    # remove depth and normal from the test sample dict
    depth_gt = test_sample.pop("depth")
    normal_gt = test_sample.pop("normal")
    # make the cam1, cam2 and mask 4 dimensional
    test_sample["cam1"] = torch.Tensor(test_sample["cam1"][None, ...])
    test_sample["cam2"] = torch.Tensor(test_sample["cam2"][None, ...])
    test_sample["mask"] = torch.Tensor(test_sample["mask"][None, ...])
    out = model.forward((test_sample["cam1"], test_sample["cam2"], test_sample["mask"]))
    print("out", out.shape)

    out_np = out.detach().cpu().numpy()
    #out_np = torch.permute(out_np, [0, 2, 3, 1])
    out_np = np.reshape(out_np, (out_np.shape[1], out_np.shape[2], out_np.shape[3]))
    depth = out_np[..., 0]
    normal = out_np[..., 1:]

    depth_tensor = torch.Tensor(depth_gt)
    near = uncompressDepth(1)
    far = uncompressDepth(0)
    d = uncompressDepth(depth_tensor)
    depth_tensor = div_no_nan(d - near, far - near)

    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).detach()
    sobel_x = sobel_x.view((1, 1, 3, 3))
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).detach()
    sobel_y = -1. * sobel_y.view((1, 1, 3, 3))

    dx = torch.nn.functional.conv2d(depth_tensor.view((1, 1, 256, 256)), sobel_x, padding=1, stride=1)
    dy = torch.nn.functional.conv2d(depth_tensor.view((1, 1, 256, 256)), sobel_y, padding=1, stride=1)

    texel_size = 1.0 / 256
    # create a tensor of ones of shape and type of out[..., 0]
    ones = torch.ones_like(dx)
    dz = ones * texel_size * 2.0

    # n = tf.concat([dx, dy, dz], -1)
    n = torch.cat([dx, dy, dz], dim=1)
    # n = normalize(n)
    print("norm", torch.norm(n, dim=1, keepdim=True).shape)
    n = n / torch.norm(n, dim=1, keepdim=True)
    n = n * 0.5 + 0.5

    n = n.squeeze(0)
    n = n.permute(1, 2, 0)
    dx = dx.squeeze(0).squeeze(0)
    dy = dy.squeeze(0).squeeze(0)

    # create a new folder Test_Results
    # and save the depth and normal maps
    if not os.path.exists("Test_Results"):
        os.makedirs("Test_Results")

    # save the depth map using matplotlib
    plt.imsave("Test_Results/depth.png", depth, cmap="gray")

    # save the normal map as rgb using matplotlib
    plt.imsave("Test_Results/normal.png", normal)

    # save the depth_gt and normal_gt using matplotlib
    plt.imsave("Test_Results/depth_gt.png", depth_gt, cmap="gray")
    plt.imsave("Test_Results/normal_gt.png", normal_gt)

    print(dx.shape)
    print(n.shape)
    print(depth_tensor.shape)
    print(depth_gt.shape)

    # save the n as rgb using matplotlib
    plt.imsave("Test_Results/n.png", n.detach().cpu().numpy())

    # save dx and dy using matplotlib
    plt.imsave("Test_Results/dx.png", dx.detach().cpu().numpy(), cmap="gray")
    plt.imsave("Test_Results/dy.png", dy.detach().cpu().numpy(), cmap="gray")

    # save depth_tensor using matplotlib
    plt.imsave("Test_Results/depth_tensor.png", depth_tensor.detach().cpu().numpy(), cmap="gray")

    print("DONE")

    # Testing:
    # trainer.test(model, test_dataloaders=data)
