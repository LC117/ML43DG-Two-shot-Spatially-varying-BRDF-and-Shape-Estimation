import pytorch_lightning as pl
import torch 
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.conv import ConvTranspose2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.activation import ReLU
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
        chn = 3
        for i in range(layers_needed):
            prev_chn = chn
            chn = min(self.base_nf * (2 ** i), 512)
            model["enc.conv{i}"] = Conv2d(
                prev_chn, 
                chn, 
                4,
                stride = 2
            )
            model["enc.maxpool{i}"] = MaxPool2d(2)
        model["ReLU"] = ReLU()
        return model
