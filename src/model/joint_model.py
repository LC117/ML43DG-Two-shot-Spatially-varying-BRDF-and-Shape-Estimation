import os

import torch
import numpy as np
import pytorch_lightning as pl

import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from math import log2
from pathlib import Path

import src.utils.rendering_layer as rl
import src.utils.sg_utils as sg
from src.data.dataloader_lightning import TwoShotBrdfDataLightning
from src.utils.common_layers import INReLU, uncompressDepth, div_no_nan, preresnet_group, preresnet_basicblock
from src.utils.merge_conv import MergeConv
from src.utils.losses import masked_loss
from src.utils.visualize_tools import save_img

MAX_VAL = 2



class JointNetwork(pl.LightningModule):
    """
    Pytorch Implementation of:
    https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/joint_network.py
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
        device = "cuda:0",
        use_gt=True
    ):
        super().__init__()
        self.base_nf = base_nf
        self.imgSize = imgSize
        self.fov = fov
        self.distance_to_zero = distance_to_zero
        
        if not torch.cuda.is_available():
            device = "cpu"

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

        self.use_gt = use_gt

        self.axis_sharpness = torch.tensor(
            sg.setup_axis_sharpness(num_sgs), dtype=torch.float32, device=device)

        # self.downscale_steps = 4
        # self.consistency_loss = 0
        # self.enable_consistency = False

        layers_needed = 3

        # env_net:
        # Define model:
        # create layers_needed conv layers with skip connections
        self.encoder_layers = []
        channel_sizes = []
        in_channels = 87 # 90  # 42
        for i in range(layers_needed):
            channel_sizes.append(in_channels)
            out_channels = min(self.base_nf * (2 ** i), 512)
            new_layer = nn.Sequential(
                # torch.nn.ZeroPad2d((1, 2, 1, 2)),
                torch.nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                INReLU(out_channels)
            )
            self.encoder_layers.append(new_layer)
            in_channels = out_channels
        # store encoder layers in a torch.nn.ModuleList
        self.encoder = torch.nn.ModuleList(self.encoder_layers)

        # resnet blocks
        print("encoder size ", in_channels)
        resnet_blocks = 4
        out_channels = 256
        self.resnet, self.resnet_end = preresnet_group(
            in_channels,
            preresnet_basicblock,
            out_channels,
            resnet_blocks,
            1,
            True,
        )
        in_channels = out_channels

        # decoder
        channel_sizes.reverse()
        #in_channels = 256
        decoder_part_layers = []
        decoder_layers = []
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
            decoder_part_layers.append(layer)
            decoder_layers.append(nn.Sequential(*decoder_part_layers))
            decoder_part_layers.clear()
            in_channels = nf

            # layer = Conv2D("conv%d" % (i + 1), layer, nf, 3, activation=INReLU)
            layer = nn.Sequential(
                torch.nn.Conv2d(
                    in_channels=in_channels + channel_sizes[i],
                    out_channels=nf,
                    kernel_size=3,
                    padding=1),
                INReLU(nf))
            decoder_part_layers.append(layer)

        layer = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=11,
                kernel_size=5,
                padding=2),
            torch.nn.Sigmoid())
        
        decoder_part_layers.append(layer)
        decoder_layers.append(nn.Sequential(*decoder_part_layers))
        decoder_part_layers.clear()
        self.decoder = nn.ModuleList(decoder_layers)
        #self.decoder = nn.Sequential(*self.decoder_layers)
        return

    def forward(self, x):
        # batch["flash"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"], batch["rerender_img"], batch["roughness"], batch["diffuse"], batch["specular"]
        flash, mask, normal, depth, sgs, rerender_img, roughness, diffuse, specular = x
        
        onesTensor = torch.ones_like(mask)
        sgs_expanded = sgs.reshape(-1, sgs.shape[1] * sgs.shape[2], 1, 1)
        sgs_to_add = onesTensor * sgs_expanded
        
        loss_img = torch.abs(flash - rerender_img) * mask
        
        # x = torch.cat([loss_img, diffuse, specular, roughness, normal, depth, mask, sgs], dim=1)

        onesTensor = torch.ones_like(mask)
        sgs_expanded = torch.reshape(
            sgs, [-1, sgs.shape[1] * sgs.shape[2], 1, 1]
        )
        sgs_to_add = onesTensor * sgs_expanded
        
        brdfInput = torch.cat(
            [loss_img, diffuse, specular, roughness, normal, depth, sgs_to_add, mask],
            axis=1
        )
        
        x = brdfInput
        
        skips = []
        for encoder_layer in self.encoder:
            # print("Encoder layer", x.shape)
            skips.append(x)
            x = encoder_layer(x)

        for i in range(0, len(self.resnet), 2):
            skip = x
            x = self.resnet[i](x)
            x += self.resnet[i + 1](skip)

        for decoder_layer in self.decoder:
            x = decoder_layer(x)
            # concat x and skip
            if len(skips) > 0:
                # print("Concat", x.shape, skips[-1].shape)
                x = torch.cat([x, skips.pop()], dim=1)

        diffuse = torch.clamp(x[:, 0:3, :, :], 0.0, 1.0)
        specular = torch.clamp(x[:, 3:6, :, :], 0.0, 1.0) * mask
        
        # Ensure energy conversation
        diffuse = (diffuse * (1 - specular)) * mask
        
        roughness = torch.clamp(x[:, 6:7, :, :], 0.004, 1.0) * mask
        
        normal = torch.clamp(x[:, 7:10, :, :], 0.0, 1.0) * mask

        # calculate the mean depth of the complete depth image
        depth = torch.clamp(x[:, 10:11, :, :], 0.0, 1.0) * mask

        return (diffuse, specular, roughness, normal, depth)

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
        
        #print("batch", batch.keys())
        append_gt = "_gt" if self.use_gt else ""
        if self.use_gt:
            x = batch["cam1"], batch["mask"], batch["normal_gt"], batch["depth_gt"], batch["sgs"], batch["cam2"],\
                batch["roughness_gt"], batch["diffuse_gt"], batch["specular_gt"]
        else:
            x = batch["flash"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"],\
                batch["rerender_img"], batch["roughness"], batch["diffuse"], batch["specular"]
        # targets = batch["normal"], batch["depth"], batch["roughness"], batch["diffuse"], batch["specular"]

        # Perform a forward pass on the network with inputs
        diffuse, specular, roughness, normal, depth = self.forward(x)
        # normal, depth, roughness, diffuse, specular = self.forward(x)

        #loss_img = torch.abs(flash_img - rerender_img)

        diffuse_loss = masked_loss(diffuse, batch["diffuse" + append_gt], mask, torch.nn.L1Loss())
        specular_loss = masked_loss(specular, batch["specular" + append_gt], mask, torch.nn.L1Loss())
        roughness_loss = masked_loss(roughness, batch["roughness" + append_gt], mask, torch.nn.L1Loss())
        normal_loss = masked_loss(normal, batch["normal" + append_gt], mask, torch.nn.L1Loss())
        depth_loss = masked_loss(depth, batch["depth" + append_gt], mask, torch.nn.L1Loss())

        if self.rendering_loss: # NOT USED IN ORIGINAL
            rendered = self.render(
                diffuse, specular, roughness, normal, depth, batch["sgs"], mask
            )

            rerendered_log = torch.nn.functional.relu(rendered)
            # clip the rerendered log between 0 and 13
            rerendered_log = torch.clamp(torch.log(1.0 + rerendered_log), 0.0, 13.0)
            
            loss_log = torch.clamp(torch.log(1.0 + batch["flash"]), 0.0, 13.0)

            rendered_loss = masked_loss(rerendered_log, loss_log, mask, torch.nn.L1Loss())


        shape_loss = diffuse_loss + specular_loss + roughness_loss + normal_loss + depth_loss

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

    train = True
    save = False
    use_gt = False
    show_model_predictions = True
    infer_mode = "overfit"
    resume_training = False
    batch_size = 8
    num_workers = 4
    epochs = 100

    overfit = infer_mode == "overfit"
    if overfit:
        infer_mode = "overfit"
        batch_size = 5
        num_workers = 0

    if not train:
        epochs = 1

    resume_from_checkpoint = None
    if resume_training:
        # check if the last to parts of the current execution path are src/model/
        execution_from_model = "src" in os.getcwd() and "model" in os.getcwd()
        prefix = "../../" if execution_from_model else ""

        path_start = Path(prefix + "lightning_logs")
        ckpt_path = Path("epoch=7-step=12375.ckpt")
        ckpt_path = path_start / "version_155" / "checkpoints" / ckpt_path
        resume_from_checkpoint = str(ckpt_path)
        model = JointNetwork.load_from_checkpoint(
            checkpoint_path=str(ckpt_path))
    else:
        model = JointNetwork(use_gt=use_gt)

    data = TwoShotBrdfDataLightning(mode="joint", overfit=overfit, num_workers=num_workers,
                                    batch_size=batch_size,
                                    persistent_workers=num_workers > 0, pin_memory=numGPUs > 0, shuffle=train,
                                    use_gt=use_gt)
    dataloaders = {
        "train": data.train_dataloader,
        "val": data.val_dataloader,
        "test": data.test_dataloader,
        "overfit": data.val_dataloader,
    }
    callbacks = []  # if train else [SavePredictionCallback(dataloaders[infer_mode](), infer_mode, batch_size)]

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=2, mode="min")
    if save or train:
        trainer = pl.Trainer(
            # weights_summary="full",
            max_epochs=epochs if numGPUs else 0,
            progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
            gpus=numGPUs,  # Use GPU if available
            profiler="simple",
            # precision=16,
            # callbacks=[early_stop_callback]
            callbacks=callbacks
        )

        if train:
            if resume_training:
                trainer.fit(model, train_dataloaders=data, ckpt_path=resume_from_checkpoint)
            else:
                trainer.fit(model, train_dataloaders=data)
        else:
            if resume_training:
                trainer.predict(
                    model, dataloaders=dataloaders[infer_mode](), ckpt_path=resume_from_checkpoint)
            else:
                trainer.predict(model, dataloaders=dataloaders[infer_mode]())
    else:
        trainer = pl.Trainer(
            # weights_summary="full",
            max_epochs=epochs if numGPUs else 0,
            progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
            gpus=numGPUs,  # Use GPU if available
            profiler="simple",
            resume_from_checkpoint=resume_from_checkpoint,
        )

    if show_model_predictions:
        batch = next(iter(data.val_dataloader()))
        print("x keys", batch.keys())
        if use_gt:
            x = batch["cam1"], batch["mask"], batch["normal_gt"], batch["depth_gt"], batch["sgs"], batch["cam2"], \
                batch["roughness_gt"], batch["diffuse_gt"], batch["specular_gt"]
        else:
            x = batch["flash"], batch["mask"], batch["normal"], batch["depth"], batch["sgs"], batch[
                "rerender_img"], \
                batch["roughness"], batch["diffuse"], batch["specular"]

        diffuse_pred, specular_pred, roughness_pred, normal_pred, depth_pred = model.forward(x)

        path_joint = "Test_Results/joint_using_gt_" + str(use_gt) + "/"
        print("Saving output to " + path_joint)

        for k in batch:
            if not "sgs" in k:
                save_img(batch[k], path_joint, k)

        save_img(normal_pred, path_joint, "normal_pred1")
        save_img(depth_pred, path_joint, "depth_pred1")
        save_img(roughness_pred, path_joint, "roughness_pred1")
        save_img(diffuse_pred, path_joint, "diffuse_pred1")
        save_img(specular_pred, path_joint, "specular_pred1")

        """
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
        test_sample["flash"] = test_sample["cam2"]  # TODO: remove this line
        #test_sample["flash"] = torch.tensor(test_sample["flash"][None, ...])
        x = test_sample["flash"], test_sample["mask"], test_sample["normal"], test_sample["depth"], test_sample["sgs"],\
            test_sample["cam1"], test_sample["roughness"], test_sample["diffuse"], test_sample["specular"]
            # test_sample["rerender_img"], test_sample["roughness"], test_sample["diffuse"], test_sample["specular"]

        normal, depth, roughness, diffuse, specular = model.forward(x)

        #out_np = out.detach().cpu().numpy()
        #out_np = torch.permute(out_np, [0, 2, 3, 1])
        #out_np = np.reshape(out_np, (out_np.shape[1], out_np.shape[2], out_np.shape[3]))
        # targets = batch["normal"], batch["depth"].unsqueeze(-1), batch["roughness"].unsqueeze(-1), batch["diffuse"], batch["specular"]
        #depth = out_np[3, ...]
        #normal = np.transpose(out_np[0:3, ...], (1, 2, 0))
        #roughness = out_np[4, ...]
        #diffuse = np.transpose(out_np[5:8, ...], (1, 2, 0))
        #specular = np.transpose(out_np[8:, ...], (1, 2, 0))

        depth = depth.detach().cpu().numpy()
        depth = depth[0, 0, ...]
        normal = normal.detach().cpu().numpy()
        normal = np.transpose(normal[0, ...], (1, 2, 0))
        roughness = roughness.detach().cpu().numpy()
        roughness = roughness[0, 0, ...]
        diffuse = diffuse.detach().cpu().numpy()
        diffuse = np.transpose(diffuse[0, ...], (1, 2, 0))
        specular = specular.detach().cpu().numpy()
        specular = np.transpose(specular[0, ...], (1, 2, 0))

        # create a new folder Test_Results
        # and save the depth and normal maps
        if not os.path.exists("Test_Results/joint"):
            os.makedirs("Test_Results/joint")

        # save the depth map using matplotlib
        plt.imsave("Test_Results/joint/depth.png", depth, cmap="gray", vmin=0, vmax=1)

        # save the normal map as rgb using matplotlib
        plt.imsave("Test_Results/joint/normal.png", normal, vmin=0, vmax=1)

        # save the depth_gt and normal_gt using matplotlib
        normal_gt = np.transpose(normal_gt[0], (1, 2, 0))
        print("depth_gt", depth_gt.shape)
        print("normal_gt", normal_gt.shape)
        plt.imsave("Test_Results/joint/depth_gt.png", depth_gt[0, 0, ...], cmap="gray", vmin=0, vmax=1)
        # convert normal_t to np array
        normal_gt = normal_gt.detach().cpu().numpy()
        plt.imsave("Test_Results/joint/normal_gt.png", normal_gt, vmin=0, vmax=1)

        # save the roughness map using matplotlib
        plt.imsave("Test_Results/joint/roughness.png", roughness, cmap="gray", vmin=0, vmax=1)

        # save the diffuse map using matplotlib
        plt.imsave("Test_Results/joint/diffuse.png", diffuse, cmap="gray", vmin=0, vmax=1)

        # save the specular map using matplotlib
        plt.imsave("Test_Results/joint/specular.png", specular[:, :, 0], cmap="gray", vmin=0, vmax=1)
        """

    print("DONE")

    # Testing:
    # trainer.test(model, test_dataloaders=data)
