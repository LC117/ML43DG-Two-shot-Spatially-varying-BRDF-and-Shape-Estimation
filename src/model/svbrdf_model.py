from pathlib import Path
import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from math import log2
from math import ceil
import numpy as np
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
from src.utils.visualize_tools import save_img


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

    def forward(self, x):
        if type(x) == tuple:
             cam1, cam2, mask, normal, depth = x
        else:
            cam1, cam2, mask, normal, depth = x["cam1"], x["cam2"], x["mask"], x["normal"], x["depth"]
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

        loss_function = L1Loss()

        loss_diffuse = loss_function(pred_diffuse, gt_diffuse)
        loss_specular = loss_function(pred_specular, gt_specular)
        loss_roughness = loss_function(pred_roughness, gt_roughness)

        if self.no_rendering_loss:
            loss = (loss_diffuse + loss_specular + loss_roughness) / 3.0
            return loss
        
        # With rendering loss
        with torch.no_grad(): # Otherwise the losses become nan!
            repeat = [1 for _ in range(len(mask.shape))]
            repeat[1] = 3
            mask3 = torch.tile(mask, repeat)

            rendered = self.render(pred_diffuse, pred_specular, pred_roughness, normal, depth, sgs, mask3)

            # Image.fromarray(np.uint8(np.transpose(rendered.detach().numpy()[0] * 255, (1, 2, 0)))).show()
            # Image.fromarray(np.uint8(np.transpose(cam1.detach().numpy()[0] * 255, (1, 2, 0)))).show()
            
            rerendered_log_pred = torch.clip(torch.log(1.0 + torch.relu(rendered)), 0.0, 13.0)
            loss_log_target = torch.clip(torch.log(1.0 + torch.relu(cam1)), 0.0, 13.0)
            
            rerendered_loss = masked_loss(rerendered_log_pred, loss_log_target, mask, loss_function)

        loss = (loss_diffuse + loss_specular + loss_roughness + rerendered_loss) / 4.0
        assert loss != torch.nan
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
        sdiff = apply_mask(diffuse, mask3, undefined=0.5)
        sspec = apply_mask(specular, mask3, undefined=0.04)
        srogh = apply_mask(roughness, mask3[:, 0:1], undefined=0.4)
        
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
        rerendered = apply_mask(rerendered, mask3)
        assert not torch.any(rerendered == torch.nan)
        return rerendered


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
        diffuse, specular, roughness = outputs

        # Create Rerender
        mask, normal, depth, sgs = batch["mask"], batch["normal"], batch["depth"], batch["sgs"]
        repeat = [1 for _ in range(len(mask.shape))]
        repeat[1] = 3
        mask3 = torch.tile(mask, repeat)

        rendered = pl_module.render(diffuse, specular, roughness, normal, depth, sgs, mask3)

        for img_id in range(diffuse.shape[0]):
            idx = batch_idx * self.batch_size + img_id
            save_dir = str(self.dataloader.dataset.gen_path(idx)) + "/"
        
            # save the images
            save_img(diffuse[img_id], save_dir, "diffuse_pred0")
            save_img(specular[img_id], save_dir, "specular_pred0")
            save_img(roughness[img_id], save_dir, "roughness_pred0")
            save_img(rendered[img_id], save_dir, "rerender0", as_exr=True)


if __name__ == "__main__":
    print("================ SV-BRDF Network ================")

    # Execution Params:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpus = 0
    if device == "cuda:0":
        gpus = 1

    batch_size = 8
    num_workers = 4
    infer_mode = "validation"
    overfit = True
    save_model = False
    test_sample = False
    train = False
    infer = True
    resume_training = True
    resume_training_version = 158
    resume_training_ckpt = "epoch=2-step=4640.ckpt"

    if overfit:
        infer_mode = "overfit"

    model = None
    if train and not resume_training:
        # Training
        model = SVBRDF_Network(device = device)
    elif (train and resume_training) or infer:
        execution_from_model = "src" in os.getcwd() and "model" in os.getcwd()
        prefix = "../../" if execution_from_model else ""
        path_start = Path(prefix + "lightning_logs")
        ckpt_path = Path(resume_training_ckpt)
        ckpt_path = path_start / f"version_{resume_training_version}" / "checkpoints" / ckpt_path
        resume_from_checkpoint = str(ckpt_path)
        model = SVBRDF_Network.load_from_checkpoint(
            checkpoint_path=str(ckpt_path))
    else:
        exit("Nothing to do! Set 'train' or 'infer' to true!")

    data = TwoShotBrdfDataLightning(mode="svbrdf", overfit=overfit, num_workers=num_workers, batch_size=batch_size, use_gt=False,
                                    shuffle=False)
    dataloaders = {
        "train": data.train_dataloader,
        "val": data.val_dataloader,
        "test": data.test_dataloader,
        "overfit": data.val_dataloader,
    }
    callbacks = [] if train else [SavePredictionCallback(dataloaders[infer_mode](), infer_mode, batch_size)]

    trainer = pl.Trainer(
        weights_summary="full",
        max_epochs=100,
        progress_bar_refresh_rate=25,
        gpus = gpus,
        profiler="simple",
        callbacks = callbacks
    )

    if train:
        pass
        #trainer.fit(model, train_dataloaders=data)
    else:
        trainer.predict(
            model, dataloaders=dataloaders[infer_mode]())

    if save_model:
        if not os.path.exists("src/trained_models/"):
            os.makedirs("src/trained_models/")
        torch.save(model.state_dict(), "src/trained_models/svbrdf_model")

    if test_sample:
        test_sample = next(iter(data.val_dataloader()))
        #test_sample = data.train_dataloader().dataset[0]
        #cam1 = torch.unsqueeze(torch.tensor(test_sample["cam1"]), dim=0)
        #cam2 = torch.unsqueeze(torch.tensor(test_sample["cam2"]), dim=0)
        #mask = torch.unsqueeze(torch.tensor(test_sample["mask"]), dim=0)
        #normal = torch.unsqueeze(torch.tensor(test_sample["normal"]), dim=0)
        #depth = torch.unsqueeze(torch.tensor(test_sample["depth"]), dim=0)

        x = test_sample
        #x = cam1, cam2, mask, normal, depth
        diffuse, specular, roughness = model.forward(x)

        #diffuse = torch.squeeze(torch.moveaxis(diffuse, 1, 3)).detach().numpy()
        #specular = torch.squeeze(torch.moveaxis(specular, 1, 3)).detach().numpy()
        #roughness = np.repeat(torch.squeeze(torch.moveaxis(roughness, 1, 3)).detach().numpy()[..., np.newaxis], 3, 2)
        results_path = "Test_Results/brdf/"
        if not os.path.exists(results_path):
            os.makedirs(results_path)
    
        #gt_diffuse = np.moveaxis(test_sample["diffuse"], 0, 2)
        #gt_specular = np.moveaxis(test_sample["specular"], 0, 2)
        #gt_roughness = np.repeat(np.moveaxis(test_sample["roughness"], 0, 2), 3, 2)
        #depth = np.repeat(np.moveaxis(test_sample["depth"], 0, 2), 3, 2)
        #normal = np.moveaxis(test_sample["normal"], 0, 2)

        # save the diffuse map as rgb using matplotlib
        save_img(diffuse, results_path, "diffuse")
        # save the specular map as rgb using matplotlib
        save_img(specular, results_path, "specular")
        # save the roughness map using matplotlib
        save_img(roughness, results_path, "roughness")

        # save ground truth
        save_img(test_sample["diffuse"], results_path, "diffuse_gt")
        save_img(test_sample["specular"], results_path, "specular_gt")
        save_img(test_sample["roughness"], results_path, "roughness_gt")
        save_img(test_sample["depth"], results_path, "depth")
        save_img(test_sample["normal"], results_path, "normal")
    
    print("DONE")


