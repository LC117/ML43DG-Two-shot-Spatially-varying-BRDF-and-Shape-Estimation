import os

import torch
import numpy as np
import pytorch_lightning as pl

import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from torch import nn

from src.data.dataloader_lightning import TwoShotBrdfDataLightning
from src.utils.common_layers import INReLU, uncompressDepth, div_no_nan, binaerize_mask
from src.utils.merge_conv import MergeConv

from pathlib import Path

from src.utils.visualize_tools import save_img

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

    def __init__(self, imgSize: int = 256, base_nf: int = 32, downscale_steps: int = 4, consistency_loss: float = 0.5,
                 device="cpu"):
        super().__init__()
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.downscale_steps = downscale_steps
        self.consistency_loss_factor = consistency_loss
        self.enable_consistency = consistency_loss != 0

        # Define model:
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

        # declare the sobelfilters as torch.FloatTensor type
        self.sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).detach()
        self.sobel_x = self.sobel_x.view((1, 1, 3, 3))
        self.sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).detach()
        self.sobel_y = - self.sobel_y.view((1, 1, 3, 3))

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

        # calculate the mean depth of the complete depth image
        depth = x[:, 3:4, :, :] * mask

        return normal, depth

    def masked_loss(self, pred, target, mask, loss):
        _mask = binaerize_mask(mask)
        # print("sum", torch.sum(_mask * (1 - _mask)), "sum", torch.sum(_mask), "sum", torch.sum(pred * _mask))
        pred = pred * _mask
        target = target * _mask
        # count how many non-zero elements are in the mask
        #n_nonzero = _mask.sum()
        loss = loss(pred, target)
        
        return loss# / n_nonzero

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
        normal_gt, depth_gt = batch["normal"], batch["depth"]

        # Perform a forward pass on the network with inputs
        normal, depth = self.forward(x)

        # In the paper the following loss is a L2 loss, but the tf implementation uses L1:
        normal_l1_loss = self.masked_loss(normal * 2 - 1, normal_gt * 2 - 1, mask, torch.nn.L1Loss())
        depth_l1_loss = self.masked_loss( depth, depth_gt, mask, torch.nn.L1Loss())
        consistency_loss = 0

        if self.enable_consistency:
            near = uncompressDepth(1)
            far = uncompressDepth(0)
            d = uncompressDepth(depth)
            depth_inv = div_no_nan(d - near, far - near)

            # calculate the gradient dx, dy and dz of the predicted depth map
            dx = F.conv2d(depth_inv, self.sobel_x, padding=1, stride=1).detach()
            dy = F.conv2d(depth_inv, self.sobel_y, padding=1, stride=1).detach()

            texel_size = 1 / self.imgSize
            # create a tensor of ones of shape and type of out[..., 0]
            dz = torch.ones_like(dx) * texel_size * 2

            cn = torch.cat([dx, dy, dz], dim=1)
            cn = cn / torch.norm(cn, dim=1, keepdim=True)
            cn = cn * 0.5 + 0.5
            
            # In the paper the following loss is a L1 loss, but the tf implementation uses L2:
            consistency_loss = self.consistency_loss_factor * \
                               self.masked_loss(cn, normal, mask, torch.nn.MSELoss())

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
    numGPUs = torch.cuda.device_count()
    device = "cuda:0" if numGPUs else "cpu"

    resume_from_checkpoint = None
    resume_training = True
    if resume_training:
        path_start = Path("../../lightning_logs")
        ckpt_path = Path("epoch=3-step=6187.ckpt")
        ckpt_path = path_start / "version_139" / "checkpoints" / ckpt_path
        resume_from_checkpoint = str(ckpt_path)
        model = ShapeNetwork.load_from_checkpoint(
            checkpoint_path=str(ckpt_path))
    else:
        model = ShapeNetwork(consistency_loss=0.5, device=device)  # downscale_steps=2, base_nf=4)

    trainer = pl.Trainer(
        #weights_summary="full",
        max_epochs=1 if numGPUs else 0,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        gpus=numGPUs,  # Use GPU if available
        profiler="simple",
        #resume_from_checkpoint=resume_from_checkpoint,
        #precision=16,
    )

    data = TwoShotBrdfDataLightning(mode="shape", overfit=False, num_workers=4, batch_size=4, persistent_workers=True, pin_memory=True)

    # trainer.fit(model, train_dataloaders=data)

    test_sample = data.val_dataloader().dataset[0]
    # remove depth and normal from the test sample dict
    depth_gt = test_sample["depth"].squeeze(0)
    normal_gt = (test_sample["normal"], (1, 2, 0))
    #depth_gt = test_sample.pop("depth").squeeze(0)
    #normal_gt = np.transpose(test_sample.pop("normal"), (1, 2, 0))
    # make the cam1, cam2 and mask 4 dimensional
    test_sample["cam1"] = torch.Tensor(test_sample["cam1"][None, ...])
    test_sample["cam2"] = torch.Tensor(test_sample["cam2"][None, ...])
    test_sample["mask"] = torch.Tensor(test_sample["mask"][None, ...])
    normal, depth = model.forward((test_sample["cam1"], test_sample["cam2"], test_sample["mask"]))

    result_dir = str(Path("Test_Results") / Path("shape_model") / Path("validation")) + "/"

    save_img(normal, result_dir, "normal")
    save_img(depth, result_dir, "depth")

    for k in test_sample.keys():
        save_img(test_sample[k], result_dir, k + "_gt")

    """
    normal = normal.permute(0, 2, 3, 1).squeeze(0)
    depth = depth.squeeze(0).squeeze(0)
    normal = normal.detach().cpu().numpy()
    depth = depth.detach().cpu().numpy()

    # create a new folder Test_Results
    # and save the depth and normal maps
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # save cam1 and cam2
    cam1 = test_sample["cam1"].squeeze(0).detach().cpu().numpy()
    cam2 = test_sample["cam2"].squeeze(0).detach().cpu().numpy()
    cam1 = np.transpose(cam1, (1, 2, 0))
    cam2 = np.transpose(cam2, (1, 2, 0))
    plt.imsave(result_dir + "cam1.png", cam1, vmin=0, vmax=1)
    plt.imsave(result_dir + "cam2.png", cam2, vmin=0, vmax=1)

    # save mask
    mask_img = test_sample["mask"][0].detach().cpu().numpy()[0, ...]
    plt.imsave(result_dir + "mask.png", mask_img, cmap="gray", vmin=0, vmax=1)

    # save the depth map using matplotlib
    plt.imsave(result_dir + "depth.png", depth, cmap="gray", vmin=0, vmax=1)

    # save the normal map as rgb using matplotlib
    plt.imsave(result_dir + "normal.png", normal, vmin=0, vmax=1)

    # save the depth_gt and normal_gt using matplotlib
    plt.imsave(result_dir + "depth_gt.png", depth_gt, cmap="gray", vmin=0, vmax=1)
    plt.imsave(result_dir + "normal_gt.png", normal_gt, vmin=0, vmax=1)
    """

    print("DONE")

    # Testing:
    # trainer.test(model, dataloaders=data)
