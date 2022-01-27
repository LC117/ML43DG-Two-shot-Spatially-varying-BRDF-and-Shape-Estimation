# -----------------------------------------------------------------------
# Copyright (c) 2020, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).
#
# Official Implementation of the CVPR2020 Paper
# Two-shot Spatially-varying BRDF and Shape Estimation
# Mark Boss, Varun Jampani, Kihwan Kim, Hendrik P. A. Lensch, Jan Kautz
# -----------------------------------------------------------------------

"""
ATTENTION THE NEW PYTORCH CODE EXPECTS ALWAYS DATA FORMAT: N,C,H,W !!! 
-> Probable reason if sth. fails !
"""
import imp
import os
import sys
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter   
summary_writer = None # instantiate only if necessary
# import tensorflow as tf
# from tensorflow.python.framework import tensor_shape
from torchvision.utils import save_image
from PIL import Image        
        
import src.utils.common_layers as cl
import src.utils.layer_helper as layer_helper
from src.utils.common_layers import (
    isclose,
    safe_sqrt,
    saturate,
    srgb_to_linear,
)
# from utils.dataflow_utils import apply_mask, chwToHwc, ensureSingleChannel, hwcToChw

from src.utils.common_layers import div_no_nan

EPS = 1e-7

class RenderingLayer(nn.Module):
    def __init__(
        self,
        fov: int,
        distanceToZero: float,
        output_shape: torch.Size,
        #data_format: str = "channels_first" # want to remove this -> "channel_fist" is default in pytoch
        data_format = "channels_first"
    ):
        """
        @ DONE 
        """
        self.distanceToZero = distanceToZero
        self.fov = fov
        # self.data_format = layer_helper.normalize_data_format(data_format)
        self.data_format = data_format  

        device = "cuda:0"
        if not torch.cuda.is_available():
            device = "cpu"
        self.device__ = device

        self.build(output_shape)

    def build(self, output_shape):
        """
        @ DONE
        """
        # Preparation
        # channel is always first:
        channel_axis = 1
        height_axis = 2
        width_axis = 3

        height = output_shape[height_axis]
        width = output_shape[width_axis]

        yRange = xRange = self.distanceToZero * np.tan((self.fov * np.pi / 180) / 2)

        x, y = np.meshgrid(
            np.linspace(-xRange, xRange, height),
            np.linspace(-yRange, yRange, width),
        )
        y = np.flipud(y)
        x = np.fliplr(x)

        z = np.ones((height, width), dtype=np.float32)
        coord = np.stack([x, y, z]).astype(np.float32)
        
        # if self.data_format == "channels_last": # if target is channel last
        #     coord = chwToHwc(coord)

        assert coord.dtype == np.float32
        self.base_mesh = torch.tensor(np.expand_dims(coord, 0), device=self.device__)
        assert self.base_mesh.dtype == torch.float32

    def call(
        self,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        camera_pos: torch.Tensor,
        light_pos: torch.Tensor,
        light_color: torch.Tensor,
        sgs: torch.Tensor,
    ) -> torch.Tensor:
        """ 
        @ DONE
        Evaluate the rendering equation
        """
        #Setup":
        assert (
            sgs.shape[self._get_channel_axis()] == 7 and len(sgs.shape) == 3
        )  # n, sgs, c
        assert (
            diffuse.shape[self._get_channel_axis()] == 3 and len(diffuse.shape) == 4
        )
        assert (
            specular.shape[self._get_channel_axis()] == 3
            and len(specular.shape) == 4
        )
        assert (
            roughness.shape[self._get_channel_axis()] == 1
            and len(roughness.shape) == 4
        )
        assert (
            normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
        )
        assert mask.shape[self._get_channel_axis()] == 1 and len(mask.shape) == 4
        assert (
            camera_pos.shape[self._get_channel_axis()] == 3
            and len(camera_pos.shape) == 2
        )
        assert (
            light_pos.shape[self._get_channel_axis()] == 3
            and len(light_pos.shape) == 2
        )
        assert (
            light_color.shape[self._get_channel_axis()] == 3
            and len(light_color.shape) == 2
        )

        realDepth = self._uncompressDepth(depth)
        perturbed_mesh = self.base_mesh * realDepth

        # Is expected to already have correct shape:
        # if self.data_format == "channels_first":
        #     reshapeShape = [-1, 3, 1, 1]
        # else:
        #     reshapeShape = [-1, 1, 1, 3]
        reshapeShape = [-1, 3, 1, 1]
        lp = torch.reshape(light_pos, reshapeShape)
        vp = torch.reshape(camera_pos, reshapeShape)
        lc = torch.reshape(light_color, reshapeShape)

        l_vec = lp - perturbed_mesh

        v = self._normalize(vp - perturbed_mesh)
        l = self._normalize(l_vec)
        h = self._normalize(l + v)

        axis_flip = torch.tensor([-1, 1, -1], dtype=torch.float32, device=self.device__)
        axis_flip = torch.reshape(axis_flip, reshapeShape)
        n = self._normalize(normal * 2.0 - 1.0) * axis_flip

        ndl = saturate(self._dot(n, l))
        ndv = saturate(self._dot(n, v), 1e-5)
        ndh = saturate(self._dot(n, h))
        ldh = saturate(self._dot(l, h))
        vdh = saturate(self._dot(v, h))

        sqrLightDistance = self._dot(l_vec, l_vec)
        light = div_no_nan(lc, sqrLightDistance)

        diff = srgb_to_linear(diffuse)
        spec = srgb_to_linear(specular)

        directSpecular = self.spec(ndl, ndv, ndh, ldh, vdh, spec, roughness)
        # Diffuse:
        directDiffuse = diff * (1.0 / np.pi) * ndv * (1.0 - self.F(spec, ldh))

        # Direct_light:
        brdf = directSpecular + directDiffuse
        direct = brdf * light

        direct = torch.where(
            torch.less(self._to_vec3(ndl), EPS), torch.zeros_like(direct), direct
        )
        direct = torch.where(
            torch.less(self._to_vec3(ndv), EPS), torch.zeros_like(direct), direct
        )

        # SG_light:
        # sg_stack_axis = 2 if self.data_format == "channels_first" else 1
        sg_stack_axis = 2
        number_of_sgs = sgs.shape[sg_stack_axis]

        env_direct = torch.zeros_like(direct)
        for i in range(number_of_sgs):
            # if self.data_format == "channels_first":
            #     sg = sgs[:, :, i]
            # else:
            #     sg = sgs[:, i]
            sg = sgs[:, :, i]
            evaled = self.sg_eval(
                sg, diffuse, specular, roughness, n, mask, perturbed_mesh, vp
            )

            env_direct = env_direct + evaled

        # Blending:
        return direct + env_direct

    def F(self, F0: torch.Tensor, ldh: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Fresnel:
        ct = 1 - ldh
        ctsq = ct * ct
        ct5 = ctsq * ctsq * ct
        return F0 + (1 - F0) * ct5

    def _G(self, a2: torch.Tensor, ndx: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        return div_no_nan(2 * ndx, ndx + safe_sqrt(a2 + (1 - a2) * ndx * ndx))

    def G(self, alpha: torch.Tensor, ndl: torch.Tensor, ndv: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Geometry:
        a2 = alpha * alpha
        return self._G(a2, ndl) * self._G(a2, ndv)

    def D(self, alpha: torch.Tensor, ndh: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Distribution:
        a2 = alpha * alpha

        denom = (ndh * ndh) * (a2 - 1) + 1.0
        denom2 = denom * denom

        return div_no_nan(a2, np.pi * denom2)

    def spec(
        self,
        ndl: torch.Tensor,
        ndv: torch.Tensor,
        ndh: torch.Tensor,
        ldh: torch.Tensor,
        vdh: torch.Tensor,
        F0: torch.Tensor,
        roughness: torch.Tensor,
    ) -> torch.Tensor:
        """
        @ DONE 
        """
        # Specular:
        alpha = saturate(roughness * roughness, 1e-3)

        F = self.F(F0, ldh)
        G = self.G(alpha, ndl, ndv)
        D = self.D(alpha, ndh)

        ret = div_no_nan(F * G * D, 4.0 * ndl)

        ret = torch.where(
            torch.less(self._to_vec3(ndh), EPS), torch.zeros_like(ret), ret
        )
        ret = torch.where(
            torch.less(self._to_vec3(ldh * ndl), EPS), torch.zeros_like(ret), ret
        )
        ret = torch.where(
            torch.less(self._to_vec3(vdh * ndv), EPS), torch.zeros_like(ret), ret
        )
        return ret

    def _sg_integral(self, sg: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Integral:
        assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4

        s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

        expTerm = 1.0 - torch.exp(-2.0 * s_sharpness)
        return 2 * np.pi * div_no_nan(s_amplitude, s_sharpness) * expTerm

    def _sg_evaluate(self, sg: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Evaluate:
        assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4
        assert d.shape[self._get_channel_axis()] == 3 and len(d.shape) == 4

        s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

        cosAngle = self._dot(d, s_axis).transpose(2, 3) # Hopefully the results are the same..
        return s_amplitude * torch.exp(s_sharpness * (cosAngle - 1.0))

    def _sg_inner_product(self, sg1: torch.Tensor, sg2: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # InnerProd:
        assert sg1.shape[self._get_channel_axis()] == 7 and len(sg1.shape) == 4
        assert sg2.shape[self._get_channel_axis()] == 7 and len(sg2.shape) == 4

        s1_amplitude, s1_axis, s1_sharpness = self._extract_sg_components(sg1)
        s2_amplitude, s2_axis, s2_sharpness = self._extract_sg_components(sg2)

        umLength = self._magnitude(s1_sharpness * s1_axis + s2_sharpness * s2_axis)
        expo = (
            torch.exp(umLength - s1_sharpness - s2_sharpness)
            * s1_amplitude
            * s2_amplitude
        )

        other = 1.0 - torch.exp(-2.0 * umLength)

        return div_no_nan(2.0 * np.pi * expo * other, umLength)

    def _sg_evaluate_diffuse(
        self, sg: torch.Tensor, diffuse: torch.Tensor, normal: torch.Tensor
    ) -> torch.Tensor:
        """
        @ DONE 
        """
        # Diffuse:
        assert (
            sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4
        )  # b, h, w, c | b, c, h, w
        assert (
            diffuse.shape[self._get_channel_axis()] == 3 and len(diffuse.shape) == 4
        )
        assert (
            normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
        )

        diff = div_no_nan(diffuse, np.pi)

        s_amplitude, s_axis, s_sharpness = self._extract_sg_components(sg)

        mudn = saturate(self._dot(s_axis, normal))

        c0 = 0.36
        c1 = 1.0 / (4.0 * c0)

        eml = torch.exp(-s_sharpness)
        em2l = eml * eml
        rl = div_no_nan(1.0, s_sharpness)

        scale = 1.0 + 2.0 * em2l - rl
        bias = (eml - em2l) * rl - em2l

        x = safe_sqrt(1.0 - scale)
        x0 = c0 * mudn
        x1 = c1 * x

        n = x0 + x1

        y = torch.where(torch.le(torch.abs(x0), x1), n * div_no_nan(n, x), mudn)

        res = scale * y + bias

        res = res * self._sg_integral(sg) * diff

        return res

    def _sg_distribution_term(self, d: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Distribution:
        assert d.shape[self._get_channel_axis()] == 3 and len(d.shape) == 4
        assert (
            roughness.shape[self._get_channel_axis()] == 1
            and len(roughness.shape) == 4
        )

        a2 = saturate(roughness * roughness, 1e-3)

        ret = torch.cat(
            [
                self._to_vec3(div_no_nan(1.0, np.pi * a2)),
                d,
                div_no_nan(2.0, a2),
            ],
            self._get_channel_axis(),
        )

        return ret

    def _sg_warp_distribution(self, ndfs: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # WarpDistribution:
        assert ndfs.shape[self._get_channel_axis()] == 7 and len(ndfs.shape) == 4
        assert v.shape[self._get_channel_axis()] == 3 and len(v.shape) == 4

        ndf_amplitude, ndf_axis, ndf_sharpness = self._extract_sg_components(ndfs)

        ret = torch.cat(
            [
                ndf_amplitude,
                self._reflect(-v, ndf_axis),
                div_no_nan(
                    ndf_sharpness, (4.0 * saturate(self._dot(ndf_axis, v), 1e-4))
                ),
            ],
            self._get_channel_axis(),
        )

        return ret

    def _sg_ggx(self, a2: torch.Tensor, ndx: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        # Geometric :
        return div_no_nan(1.0, (ndx + safe_sqrt(a2 + (1 - a2) * ndx * ndx)))

    def _sg_evaluate_specular(
        self,
        sg: torch.Tensor,
        specular: torch.Tensor,
        roughness: torch.Tensor,
        warped_ndf: torch.Tensor,
        ndl: torch.Tensor,
        ndv: torch.Tensor,
        ldh: torch.Tensor,
        vdh: torch.Tensor,
        ndh: torch.Tensor,
    ) -> torch.Tensor:
        """
        @ DONE 
        """
        # Specular:
        assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 4
        assert (
            warped_ndf.shape[self._get_channel_axis()] == 7
            and len(warped_ndf.shape) == 4
        )
        assert (
            specular.shape[self._get_channel_axis()] == 3
            and len(specular.shape) == 4
        )
        assert (
            roughness.shape[self._get_channel_axis()] == 1
            and len(roughness.shape) == 4
        )
        assert ndl.shape[self._get_channel_axis()] == 1 and len(ndl.shape) == 4
        assert ndv.shape[self._get_channel_axis()] == 1 and len(ndv.shape) == 4
        assert ldh.shape[self._get_channel_axis()] == 1 and len(ldh.shape) == 4
        assert vdh.shape[self._get_channel_axis()] == 1 and len(vdh.shape) == 4
        assert ndh.shape[self._get_channel_axis()] == 1 and len(ndh.shape) == 4

        a2 = saturate(roughness * roughness, 1e-3)

        # Distribution :
        Distribution = self._sg_inner_product(warped_ndf, sg)

        G = self._sg_ggx(a2, ndl) * self._sg_ggx(a2, ndv)

        # Fresnel:
        powTerm = torch.pow(1.0 - ldh, 5)
        Fresnel = specular + (1.0 - specular) * powTerm

        output = Distribution * G * Fresnel * ndl

        shadowed = torch.zeros_like(output)
        zero_vec = torch.zeros_like(ndh)
        output = torch.where(self._to_vec3(isclose(ndh, zero_vec)), shadowed, output)
        output = torch.where(
            self._to_vec3(isclose(ldh * ndl, zero_vec)), shadowed, output
        )
        output = torch.where(
            self._to_vec3(isclose(vdh * ndv, zero_vec)), shadowed, output
        )
        return torch.maximum(output, torch.zeros(output.shape, device=self.device__))

    def sg_eval(
        self,
        sg: torch.Tensor,
        diffuse: torch.Tensor,
        specular: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,
        mask: torch.Tensor,
        perturbed_mesh: torch.Tensor,
        camera_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        @ DONE 
        """
        # SG: (Spherical Gaussian)
        assert sg.shape[self._get_channel_axis()] == 7 and len(sg.shape) == 2
        assert (
            diffuse.shape[self._get_channel_axis()] == 3 and len(diffuse.shape) == 4
        )
        assert (
            specular.shape[self._get_channel_axis()] == 3
            and len(specular.shape) == 4
        )
        assert (
            roughness.shape[self._get_channel_axis()] == 1
            and len(roughness.shape) == 4
        )
        assert (
            normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
        )
        assert (
            normal.shape[self._get_channel_axis()] == 3 and len(normal.shape) == 4
        )
        assert mask.shape[self._get_channel_axis()] == 1 and len(mask.shape) == 4
        assert (
            camera_pos.shape[self._get_channel_axis()] == 3
            and len(camera_pos.shape) == 4
        )

        # if self.data_format == "channels_first":
        #     sgShape = [-1, 7, 1, 1]
        # else:
        #     sgShape = [-1, 1, 1, 7]
        sgShape = [-1, 7, 1, 1]
        sg = torch.reshape(sg, sgShape)

        v = self._normalize(camera_pos - perturbed_mesh)
        diff = srgb_to_linear(diffuse)
        spec = srgb_to_linear(specular)
        norm = normal
        rogh = roughness

        ndf = self._sg_distribution_term(norm, rogh)

        warped_ndf = self._sg_warp_distribution(ndf, v)
        _, wndf_axis, _ = self._extract_sg_components(warped_ndf)

        warpDir = wndf_axis

        ndl = saturate(self._dot(norm, warpDir))
        ndv = saturate(self._dot(norm, v), 1e-5)
        h = self._normalize(warpDir + v)
        ndh = saturate(self._dot(norm, h))
        ldh = saturate(self._dot(warpDir, h))
        vdh = saturate(self._dot(v, h))

        diffuse_eval = self._sg_evaluate_diffuse(sg, diff, norm) * ndl
        specular_eval = self._sg_evaluate_specular(
            sg, spec, rogh, warped_ndf, ndl, ndv, ldh, vdh, ndh
        )

        shadowed = torch.zeros_like(diffuse_eval)
        zero_vec = torch.zeros_like(ndl)
        diffuse_eval = torch.where(
            self._to_vec3(isclose(ndl, zero_vec)), shadowed, diffuse_eval
        )
        diffuse_eval = torch.where(
            self._to_vec3(isclose(ndv, zero_vec)), shadowed, diffuse_eval
        )

        specular_eval = torch.where(
            self._to_vec3(isclose(ndl, zero_vec)), shadowed, specular_eval
        )
        specular_eval = torch.where(
            self._to_vec3(isclose(ndv, zero_vec)), shadowed, specular_eval
        )

        brdf_eval = diffuse_eval + specular_eval

        return torch.where(
            self._to_vec3(torch.eq(mask, zero_vec)), shadowed, brdf_eval
        )

    def _extract_sg_components(
        self, sg: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        @ DONE 
        """
        # if self.data_format == "channels_first":
        #     s_amplitude = sg[:, 0:3]
        #     s_axis = sg[:, 3:6]
        #     s_sharpness = sg[:, 6:7]
        # else:
        #     s_amplitude = sg[..., 0:3]
        #     s_axis = sg[..., 3:6]
        #     s_sharpness = sg[..., 6:7]
        
        s_amplitude = sg[:, 0:3]
        s_axis = sg[:, 3:6]
        s_sharpness = sg[:, 6:7]

        return (s_amplitude, s_axis, s_sharpness)

    def visualize_sgs(self, sgs: torch.Tensor, output: torch.Tensor, name: str = "sgs"):
        """
        @ DONE 
        """
        # Visualize:
        us, vs = torch.meshgrid(
            torch.linspace(0.0, 1.0, output.shape[3]),
            torch.linspace(0.0, 1.0, output.shape[2]), # values in shape[] increased to match channel first
        )  # OK

        uvs = torch.stack([us, vs], -1)
        # q   f

        theta = 2.0 * np.pi * uvs[..., 0] - (np.pi / 2)
        phi = np.pi * uvs[..., 1]

        d = torch.stack(
                [torch.cos(theta) * torch.sin(phi),
                 torch.cos(phi),
                 torch.sin(theta) * torch.sin(phi)], axis = 0).unsqueeze(0)

        for i in range(sgs.shape[1]): # scheint zu passen 
            output = output + self._sg_evaluate(
                torch.reshape(sgs[:, i], [-1, 7, 1, 1]), d
            )
        
        # global summary_writer 
        
        # if not summary_writer: # is None
        #     summary_writer = SummaryWriter()
        # summary_writer.add_images(name, output[:1, ...])
        return output
        
    def _magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        return cl.magnitude(x, data_format=self.data_format)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        return cl.normalize(x, data_format=self.data_format)

    def _dot(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        return cl.dot(x, y, data_format=self.data_format)

    def _reflect(self, d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        return d - 2 * self._dot(d, n) * n

    def _to_vec3(self, x: torch.Tensor) -> torch.Tensor:
        """
        @ DONE 
        """
        return cl.to_vec3(x, data_format=self.data_format)

    def _get_channel_axis(self) -> int:
        """
        @ DONE 
        """
        return cl.get_channel_axis(data_format=self.data_format) # default for pytorch!

    def _uncompressDepth(
        self, d: torch.Tensor, sigma: float = 2.5, epsilon: float = 0.7
    ) -> torch.Tensor:
        """
        @ DONE
        From 0-1 values to full depth range. The possible depth range
        is modelled by sigma and epsilon and with sigma=2.5 and epsilon=0.7
        it is between 0.17 and 1.4.
        """
        return div_no_nan(1.0, 2.0 * sigma * d + epsilon)


if __name__ == "__main__":
    model =  RenderingLayer(60, 0.7, torch.Size([32, 3, 256, 256]))
    ytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(ytorch_total_params)