from typing import Any

import torch
import numpy as np
import pytorch_lightning as pl


import os
from pathlib import Path
import matplotlib.pyplot as plt
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from math import log2

import src.utils.rendering_layer as rl
import src.utils.sg_utils as sg
from src.data.dataloader_lightning import TwoShotBrdfDataLightning
from src.utils.common_layers import INReLU
from src.utils.visualize_tools import save

from pathlib import Path
import os

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

    def __init__(self, imgSize: int = 256, base_nf: int = 16, num_sgs: int = 24, log_images=False):
        super().__init__()
        self.imgSize = imgSize
        self.base_nf = base_nf
        self.num_sgs = num_sgs  # spherical gaussian
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.axis_sharpness = torch.tensor(sg.setup_axis_sharpness(num_sgs), dtype=torch.float32, device=device)
        self.renderer = None
        self.log_images = log_images
        # env_net:
        # Define model:
        layers_needed = int(log2(256) - 2)  # 256 = cam1.shape[1].value

        # enc: 
        enc_conv2d_list = []
        in_channels = 11
        for i in range(layers_needed):
            out_channels = min(self.base_nf * (2 ** i), 256)
            enc_conv2d_list.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                )
            )
            enc_conv2d_list.append(INReLU(out_channels))
            # Set in_channels for the next layer:
            in_channels = out_channels

        self.enc_conv2d_block = torch.nn.Sequential(*enc_conv2d_list)

        # env_map:
        outputSize = self.num_sgs * 3
        self.env_map = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                in_features=512,
                out_features=256
            ),
            nn.ReLU(),
            nn.Dropout(p=0.75),
            nn.Linear(
                in_features=256,
                out_features=outputSize
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if type(x) == tuple:
            cam1, cam2, mask, normal, depth = x
        else:
            cam1, cam2, mask, normal, depth = x["cam1"], x["cam2"], x["mask"], x["normal_pred"], x["depth_pred"]

        x = torch.cat([cam1, cam2, mask, normal, depth], dim=1)  # shape: (None, 256, 256, 11) = (None, 256, 256, 3+3+1+3+1)

        x = self.enc_conv2d_block(x)
        x = self.env_map(x)
        
        sgs = (x * MAX_VAL).view(-1, self.num_sgs, 3)

        return sgs  # sphericalGaussainsShape

    def configure_optimizers(self):
        # half learning rate after half of the epochs: 
        # see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/models/illumination_network.py#L238
        # and https://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ScheduledHyperParamSetter
        model_params = list(self.enc_conv2d_block.parameters()) + list(self.env_map.parameters())
        optimizer = torch.optim.Adam(model_params, lr=0.0002, betas=(0.5, 0.999))
        step_size = self.trainer.max_epochs // 2
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size if step_size > 0 else 1,
            gamma=0.5)
        return [optimizer], [scheduler]

    def general_step(self, batch, batch_idx):
        # cam1, cam2, mask, normal, depth
        x = batch["cam1"], batch["cam2"], batch["mask"], batch["normal_pred"], batch["depth_pred"]
        targets = batch["sgs"]

        # Perform a forward pass on the network with inputs
        out = self.forward(x)

        # sgs_prep:
        batch_size = out.shape[0]
        
        axis_sharpness = torch.tile(self.axis_sharpness[None, ...], (batch_size, 1, 1))
        sgs_joined = torch.cat([out, axis_sharpness], dim=-1)

        # if self.training:
        sgs_gt = torch.clip(targets, min=0.0, max=MAX_VAL)
        sgs_gt_joined = torch.cat([sgs_gt, axis_sharpness], dim=-1)

        # loss:
        # sgs:
        sgs_loss = torch.nn.MSELoss()(sgs_joined, sgs_gt_joined)
        
        # vis:
        if self.log_images:
            self.set_renderer(batch_size)
            with torch.no_grad():
                sg_output = torch.zeros([batch_size, 3, 256, 512])
                rendered_images = self.renderer.visualize_sgs(sgs_joined, sg_output)
            
            self.trainer.logger.experiment.add_image("SGS-Render", rendered_images, 0, dataformats="NCHW")
        
        return sgs_loss

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode] for x in outputs]).mean()
        return avg_loss

    def training_step(self, batch, batch_idx):
        sgs_loss = self.general_step(batch, batch_idx)
        self.log("my_loss", sgs_loss, logger=True, on_step=True, on_epoch=True)

        # batch_size = batch["cam_1"].shape[0]
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
    
    def set_renderer(self, batch_size):
        if self.renderer is None:
            self.renderer = rl.RenderingLayer(60, 0.7, torch.Size([batch_size, 3, 256, 256]))
            
    def render_and_store(self, sgs, path):
        sgs_joined = torch.cat([sgs, self.axis_sharpness.detach().cpu()], dim=-1)
        renderer = rl.RenderingLayer(60, 0.7, torch.Size([1, 3, 256, 256]))
        sg_output = torch.zeros([1, 3, 256, 512])
        rendered_images = renderer.visualize_sgs(sgs_joined[None, ...], sg_output)

        rendered_images_norm = np.transpose(rendered_images.detach().numpy()[0], (1,2,0))
        rendered_images_norm /= np.max(rendered_images_norm)
        
        plt.imsave(path, rendered_images_norm, vmin=0., vmax=1.)

class SavePredictionCallback(Callback):
    def __init__(self, dataloader, mode, batch_size):
        super().__init__()
        self.dataloader = dataloader
        self.mode = mode
        self.batch_size = batch_size

    def on_predict_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int,
    ) -> None:
        """Called when the train batch ends."""
        sgs = outputs

        #debug = str(type(normal.shape))
        # store debug in a text file
        #with open("debug.txt", "w") as f:
        #    f.write(debug)

        for img_id in range(sgs.shape[0]):
            idx = batch_idx * self.batch_size + img_id
            save_dir = str(self.dataloader.dataset.gen_path(idx)) + "/"

            # save the sgs
            save(sgs[img_id].detach().cpu().numpy(), save_dir + "sgs_pred.npy")


if __name__ == "__main__":
    # Training:
    """
    model = IlluminationNetwork()
    epochs = 1000
    trainer = pl.Trainer(
        # weights_summary="full",
        max_epochs=epochs,
        progress_bar_refresh_rate=25,  # to prevent notebook crashes in Google Colab environments
        gpus=1 if torch.cuda.is_available() else 0, # Use GPU if available
        profiler="simple",
    )

    data = TwoShotBrdfDataLightning(mode="illumination", overfit=True, num_workers=0)

    trainer.fit(model, train_dataloaders=data)
    """

    numGPUs = torch.cuda.device_count()
    device = "cuda:0" if numGPUs else "cpu"

    train = True
    save = False
    infer_mode = "train"
    resume_training = False
    batch_size = 8
    num_workers = 4
    epochs = 200

    overfit = infer_mode == "overfit"
    if overfit:
        infer_mode = "overfit"
        batch_size = 5

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
        model = IlluminationNetwork.load_from_checkpoint(
            checkpoint_path=str(ckpt_path))
    else:
        model = IlluminationNetwork()

    data = TwoShotBrdfDataLightning(mode="illumination", overfit=overfit, num_workers=num_workers, batch_size=batch_size,
                                    persistent_workers=num_workers > 0, pin_memory=numGPUs > 0, shuffle=train, use_gt=False)
    dataloaders = {
        "train": data.train_dataloader,
        "val": data.val_dataloader,
        "test": data.test_dataloader,
        "overfit": data.val_dataloader,
    }
    callbacks = [] if train else [SavePredictionCallback(dataloaders[infer_mode](), infer_mode, batch_size)]

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

    show_model_predictions = True
    if show_model_predictions:
        batch = next(iter(data.train_dataloader()))
        x = batch["cam1"], batch["cam2"], batch["mask"], batch["normal_pred"], batch["depth_pred"]
        targets = batch["sgs"]
        predictions = model.forward(x)

        print("Truths:")
        print(targets[0])
        print("Prediction:")
        print(predictions[0])


        result_dir = str(Path("Test_Results") / Path("illumination_model")) + "/"

        # create a new folder Test_Results
        # and save the depth and normal maps
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        for i in range(3):
            # Testing:
            # trainer.test(ckpt_path="best")


            sgs_joined = torch.cat([targets[i], model.axis_sharpness.detach().cpu()], dim=-1)
            renderer = rl.RenderingLayer(60, 0.7, torch.Size([1, 3, 256, 256]))
            sg_output = torch.zeros([1, 3, 256, 512])
            rendered_images = renderer.visualize_sgs(sgs_joined[None, ...], sg_output)

            rendered_images_norm = np.transpose(rendered_images.detach().numpy()[0], (1,2,0))
            rendered_images_norm /= np.max(rendered_images_norm)

            # plt.imsave(result_dir + "sgs_gt.png", np.uint8(rendered_images_norm * 255 ))
            plt.imsave(result_dir + f"sgs_gt_{i}.png", rendered_images_norm, vmin=0., vmax=1. )

        for i in range(5):
            IlluminationNetwork.render_and_store(predictions[i], path=result_dir + f"sgs_{i}_epochs_{epochs}.png")
            # 3D Plot:
            # plot the predictions and ground truth in 3d:
            # create a new plot
            fig = plt.figure()
            # set the title to "targets"
            fig.suptitle("targets r vs predictions b")
            # create a 3d plot of the targets
            ax = fig.add_subplot(111, projection='3d')
            # plot the targets
            ax.scatter(targets[i, :, 0], targets[i, :,  1], targets[i, :, 2], c='r', marker='o')
            # present the plot
            #plt.show()

            predictions_detached = predictions.detach().numpy()

            # set the title to "predictions"
            #fig.suptitle("predictions")
            # create a 3d plot of the predictions
            #ax = fig.add_subplot(111, projection='3d')
            # plot the predictions
            ax.scatter(predictions_detached[i, :, 0], predictions_detached[i, :, 1], predictions_detached[i, :, 2], c='b', marker='o')
            # present the plot
            plt.savefig(result_dir + f"3s_sgs_gt_comparison_{i}_epochs_{epochs}.png", bbox_inches='tight' )
