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


class JointNetwork(pl.LightningModule):
    """
    Pytorch Implementation of:
    https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py
    """

    def __init__(
        self,
        base_nf: int = 64,
        imgSize: int = 256,
        fov: int = 60,
        distance_to_zero: float = 0.7,
        camera_pos=np.asarray([0, 0, 0]),
        light_pos=np.asarray([0, 0, 0]),
        light_color=np.asarray([1, 1, 1]),
        light_intensity_lumen=45,
        num_sgs=24,
        rendering_loss: bool = False,
        device="cuda:0"
    ):
        super().__init__()
        self.base_nf = base_nf
        self.imgSize = imgSize
        self.fov = fov
        self.distance_to_zero = distance_to_zero

        self.camera_pos = torch.tensor(
            camera_pos.reshape([1, 3]), dtype=torch.float32, device=device)

        self.light_pos = torch.tensor(
            light_pos.reshape([1, 3]), dtype=torch.float32, device=device)

        intensity = light_intensity_lumen / (4.0 * np.pi)
        light_col = light_color * intensity
        self.light_col = torch.tensor(
            light_col.reshape([1, 3]), dtype=torch.float32, device=device)

        self.num_sgs = num_sgs
        self.rendering_loss = rendering_loss

        self.axis_sharpness = torch.tensor(
            sg.setup_axis_sharpness(num_sgs), dtype=torch.float32, device=device)

        self.downscale_steps = 4
        self.consistency_loss = 0
        self.enable_consistency = False

        layers_needed = 3

        # env_net:
        # Define model:
        #layers_needed = int(log2(256) - 2)  # 256 = cam1.shape[1].value

        """
        l = brdfInput
                    skips = []
                    for i in range(layers_needed):
                        skips.append(l)
                        l = Conv2D(
                            "conv%d" % (i + 1),
                            l,
                            min(self.base_nf * (2 ** i), 512),
                            4,
                            strides=2,
                            activation=INReLU,
                        )
        """
        # create layers_needed conv layers with skip connections
        self.encoder_layers = []
        in_channels = 90  # 42
        for i in range(layers_needed):
            out_channels = min(self.base_nf * (2 ** (i + 1)), 512)
            new_layer = nn.Sequential(
                torch.nn.ZeroPad2d((1, 2, 1, 2)),
                #torch.nn.ZeroPad2d((0, 1, 0, 1)),
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=0
                ),
                INReLU(out_channels)
            )
            self.encoder_layers.append(new_layer)
            in_channels = out_channels
        self.encoder = nn.Sequential(*self.encoder_layers)

        # resnet blocks
        """
        resnet_blocks = 4
                l = preresnet_group(
                    "resnet_blocks",
                    l,
                    preresnet_basicblock,
                    256,
                    resnet_blocks,
                    1,
                    True,
                )
        """

        resnet_blocks = 4
        self.resnet_blocks = []
        self.resnet = nn.Sequential(

        )

        # decoder
        """
        for i in range(layers_needed):
            with tf.variable_scope("up%d" % (i + 1)):
                inv_i = layers_needed - i
                nf = min(self.base_nf * (2 ** (inv_i - 1)), 512)

                l = Conv2DTranspose(
                    "tconv%d" % (i + 1),
                    l,
                    nf,
                    4,
                    strides=2,
                    activation=INReLU,
                )
                l = tf.concat(
                    [l, skips[inv_i - 1]], -1, name="skip%d" % (i + 1)
                )
                l = Conv2D("conv%d" % (i + 1), l, nf, 3, activation=INReLU)

        params = Conv2D("output", l, 11, 5, activation=tf.nn.sigmoid)
        """

        self.decoder_layers = []
        for i in range(layers_needed):
            inv_i = layers_needed - i
            nf = min(self.base_nf * (2 ** (inv_i - 1)), 512)
            layer = nn.Sequential(
                # torch.nn.ZeroPad2d((1, 2, 1, 2)),
                #torch.nn.ZeroPad2d((0, 1, 0, 1)),
                torch.nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=nf,
                    kernel_size=4,
                    stride=2,
                    padding=1),
                INReLU(nf))
            self.decoder_layers.append(layer)
            in_channels = nf

            # layer = Conv2D("conv%d" % (i + 1), layer, nf, 3, activation=INReLU)
            layer = nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=nf,
                    kernel_size=3,
                    padding=1),
                INReLU(nf))
            self.decoder_layers.append(layer)

        layer = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=11,
                kernel_size=5,
                padding=2),
            torch.nn.Sigmoid())
        self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)

        """
        # encoder:
        encoder_layers = []
        # cam1 + mask + normal + depth + sgs + rerender_img + roughness + diffuse + specular
        # 3 + 1 + 3 + 1 + 24 + 3 + 1 + 3 + 3 = 42
        inp_channels = 4  # 42
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
            torch.nn.Conv2d(inp_channels, 11, 5, padding='same'),
            torch.nn.Sigmoid()
        )
        """
        return

    def forward(self, x):
        flash, mask, normal, depth, sgs, rerender_img, roughness, diffuse, specular = x
        onesTensor = torch.ones_like(mask)
        sgs_expanded = sgs.reshape(-1, sgs.shape[1] * sgs.shape[2], 1, 1)
        sgs_to_add = onesTensor * sgs_expanded
        #print("Shapes", flash.shape, mask.unsqueeze_(3).shape, normal.shape, depth.unsqueeze_(3).shape, sgs_to_add.shape,
        #      rerender_img.shape, roughness.unsqueeze_(3).shape, diffuse.shape, specular.shape)
        x = torch.cat([flash, mask, normal, depth, sgs_to_add, rerender_img, roughness, diffuse, specular], dim=1)
        #x = torch.permute(x, (0, 3, 1, 2))
        print("x shape", x.shape)

        #_mask = mask#.clone()
        #_mask.unsqueeze_(1)

        ##cam1_cf = torch.permute(flash, (0, 3, 1, 2))
        ##cam2_cf = torch.permute(rerender_img, (0, 3, 1, 2))


        #cam1_cf = torch.cat((cam1_cf, _mask), dim=1)
        #cam2_cf = torch.cat((cam2_cf, _mask), dim=1)

        #mask_cf = torch.permute(mask, (0, 3, 1, 2))

        #x = torch.cat([cam1, cam2, mask[..., None]],
        #              dim=-1)  # shape: (None, 256, 256, 7) = (None, 256, 256, 3+3+1)
        #x = torch.permute(x, (0, 3, 1, 2))  # torch expects "channel_first" order

        #print("X shape", x.shape)
        #out = self.model_conv(x)

        #x = x.reshape(-1, x.shape[1] * x.shape[2] * x.shape[3], 1, 1)  # shape: (None, 256*256*7)
        #out = self.model(x)
        #out = out.reshape(-1, self.imgSize, self.imgSize, 4)

        x = self.encoder(x)
        print("x shape", x.shape)
        x = self.decoder(x)
        print("x shape", x.shape)

        x = x * mask# + (1 - _mask) * torch.ones_like(x)

        x_n = x[:, 0:3, :, :] * 2 - 1
        x_normalized = x_n / torch.norm(x_n, dim=1, keepdim=True)
        x[:, 0:3, :, :] = x_normalized * 0.5 + 0.5

        return x
        #return cam1#, mask

    def masked_loss(self, pred, target, mask, loss):
        _mask = mask
        pred = pred * _mask
        target = target * _mask
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
        #x = batch["cam1"], batch["cam2"], batch["mask"]
        # targets = batch["normal"], batch["depth"].reshape(-1, self.imgSize, self.imgSize, 1)
        #normal_gt, depth_gt = batch["normal"], batch["depth"]#.unsqueeze(-1)

        # x = cam1 + mask + normal + depth + sgs + rerender_img + roughness + diffuse + specular
        batch["rerender_img"] = batch["cam1"]  # TODO: remove this line
        x = batch["flash"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"], batch["rerender_img"], batch["roughness"], batch["diffuse"], batch["specular"]
        targets = batch["normal"], batch["depth"], batch["roughness"], batch["diffuse"], batch["specular"]

        # Perform a forward pass on the network with inputs
        out = self.forward(x)

        #targets = normal_gt, depth_gt.unsqueeze(-1)
        targets_joined = torch.cat(targets, dim=1)

        #print("out shape", out.shape, "targets shape", targets_joined.shape)

        """
        #print("out shape", out.shape, mask.shape)
        depth_l2_loss = self.masked_loss(out[..., 0], depth_gt, mask, torch.nn.L1Loss())
        # calulate the angle between the normal and the predicted normal
        #normal_angle_loss = 1 - torch.nn.CosineSimilarity()(out[..., 1:], targets_joined[..., 1:])
        #normal_angle_loss = torch.nn.L1Loss()(targets_joined[..., 0] * 2 - 1, out[..., 0] * 2 - 1)
        normal_angle_loss = self.masked_loss(out[..., 1:] * 2 - 1, normal_gt * 2 - 1, mask3, torch.nn.L1Loss())
        #normal_l2_loss = torch.nn.MSELoss()(out[..., 1:], batch["normal"])
        #normals_depth_consistency_loss =

        near = uncompressDepth(1)
        far = uncompressDepth(0)
        d = uncompressDepth(out[..., 0])
        depth_inv = div_no_nan(d - near, far - near)

        # calculate the gradient dx, dy and dz of the predicted depth map
        # declare the sobelfilters as torch.FloatTensor type
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=self.device).detach()
        sobel_x = sobel_x.view((1, 1, 3, 3))
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=self.device).detach()
        sobel_y = -1. * sobel_y.view((1, 1, 3, 3))
        #sobel_x_3d = torch.tensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                           [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        #                           [[1, 0, -1], [2, 0, -2], [1, 0, -1]]])
        #sobel_z_3d = torch.tensor([[[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        #                           [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        #                           [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]])
        dx = torch.nn.functional.conv2d(depth_inv.unsqueeze(1), sobel_x, padding=1, stride=1)
        dy = torch.nn.functional.conv2d(depth_inv.unsqueeze(1), sobel_y, padding=1, stride=1, )

        #filter_x = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        #filter_y = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        #filter_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        #filter_y.weight = nn.Parameter(sobel_y, requires_grad=False)
        #dx = filter_x(out[..., 0])
        #dy = filter_y(out[..., 0])

        #sobel = torch.nn.Sobel(padding_mode='zeros')
        #dx = sobel(out[..., 1])
        #dy = sobel(out[..., 1], 1)
        texel_size = 1.0 / self.imgSize
        # create a tensor of ones of shape and type of out[..., 0]
        ones = torch.ones_like(dx)
        dz = ones * texel_size * 2.0

        # n = tf.concat([dx, dy, dz], -1)
        n = torch.cat([dx, dy, dz], dim=1)
        # n = normalize(n)
        n = n / torch.norm(n, dim=1, keepdim=True)
        n = n * 0.5 + 0.5

        #consistency_loss = torch.nn.L1Loss()(n, out[..., 1:].reshape(-1, 3, self.imgSize, self.imgSize))
        consistency_loss = self.masked_loss(n.permute((0, 2, 3, 1)), out[..., 1:], mask3, torch.nn.MSELoss())

        shape_loss = depth_l2_loss + normal_angle_loss + self.consistency_loss * consistency_loss
        """

        #shape_loss = torch.nn.MSELoss()(out, targets_joined)
        shape_loss = self.masked_loss(out, targets_joined, mask, torch.nn.MSELoss())

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

    model = JointNetwork()

    trainer = pl.Trainer(
        # weights_summary="full",
        max_epochs=5,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        gpus=1,  # Use GPU if available
        profiler="simple",
        #precision=16,
    )

    data = TwoShotBrdfDataLightning(mode="all", overfit=True, num_workers=4, batch_size=8) #, persistent_workers=True, pin_memory=True)

    trainer.fit(model, train_dataloaders=data)

    test_sample = data.train_dataloader().dataset[1]
    print("test sample", test_sample.keys(), test_sample["mask"].shape)
    # remove depth and normal from the test sample dict
    depth_gt = torch.tensor(test_sample["depth"][None, ...])
    normal_gt = torch.tensor(test_sample["normal"][None, ...])
    test_sample["depth"] = torch.tensor(test_sample["depth"][None, ...])
    test_sample["normal"] = torch.tensor(test_sample["normal"][None, ...])
    # make the cam1, cam2 and mask 4 dimensional
    test_sample["cam1"] = torch.tensor(test_sample["cam1"][None, ...])
    test_sample["cam2"] = torch.tensor(test_sample["cam2"][None, ...])
    test_sample["mask"] = torch.tensor(test_sample["mask"][None, ...])
    # x = batch["flash"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"], batch["rerender_img"], batch["roughness"], batch["diffuse"], batch["specular"]
    test_sample["roughness"] = torch.tensor(test_sample["roughness"][None, ...])
    test_sample["diffuse"] = torch.tensor(test_sample["diffuse"][None, ...])
    test_sample["specular"] = torch.tensor(test_sample["specular"][None, ...])
    test_sample["sgs"] = torch.tensor(test_sample["sgs"][None, ...])
    #test_sample["rerender_img"] = torch.Tensor(test_sample["rerender_img"][None, ...])
    test_sample["flash"] = torch.tensor(test_sample["flash"][None, ...])
    x = test_sample["flash"], test_sample["mask"], test_sample["normal"], test_sample["depth"], test_sample["sgs"],\
        test_sample["cam1"], test_sample["roughness"], test_sample["diffuse"], test_sample["specular"]
        # test_sample["rerender_img"], test_sample["roughness"], test_sample["diffuse"], test_sample["specular"]

    out = model.forward(x)
    print("out", out.shape)

    out_np = out.detach().cpu().numpy()
    #out_np = torch.permute(out_np, [0, 2, 3, 1])
    out_np = np.reshape(out_np, (out_np.shape[1], out_np.shape[2], out_np.shape[3]))
    # targets = batch["normal"], batch["depth"].unsqueeze(-1), batch["roughness"].unsqueeze(-1), batch["diffuse"], batch["specular"]
    depth = out_np[4, ...],
    normal = np.transpose(out_np[0:3, ...], (1, 2, 0))
    roughness = out_np[4, ...]
    diffuse = np.transpose(out_np[5:8, ...], (1, 2, 0))
    specular = np.transpose(out_np[8:, ...], (1, 2, 0))

    # create a new folder Test_Results
    # and save the depth and normal maps
    if not os.path.exists("Test_Results/joint"):
        os.makedirs("Test_Results/joint")

    # save the depth map using matplotlib
    plt.imsave("Test_Results/joint/depth.png", depth, cmap="gray")

    # save the normal map as rgb using matplotlib
    plt.imsave("Test_Results/joint/normal.png", normal)

    # save the depth_gt and normal_gt using matplotlib
    plt.imsave("Test_Results/joint/depth_gt.png", depth_gt, cmap="gray")
    plt.imsave("Test_Results/joint/normal_gt.png", normal_gt)

    # save the roughness map using matplotlib
    plt.imsave("Test_Results/joint/roughness.png", roughness, cmap="gray")

    # save the diffuse map using matplotlib
    plt.imsave("Test_Results/joint/diffuse.png", diffuse, cmap="gray")

    # save the specular map using matplotlib
    plt.imsave("Test_Results/joint/specular.png", specular, cmap="gray")


    print("DONE")

    # Testing:
    # trainer.test(model, test_dataloaders=data)
