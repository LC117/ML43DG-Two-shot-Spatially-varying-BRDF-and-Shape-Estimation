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
    Same as: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/utils/dataflow_utils.py
"""
import os

import cv2
import numpy as np
import pyexr
from skimage.morphology import disk, erosion


from src.utils.config import *


# def apply_mask(
#     img: np.ndarray, mask: np.ndarray, undefined: np.ndarray = np.asarray([0, 0, 0])
# ):
#     return np.where(mask == 1, img, undefined)


def hwcToChw(hwc: np.ndarray) -> np.ndarray:
    return np.moveaxis(hwc, -1, 0)


def chwToHwc(chw: np.ndarray) -> np.ndarray:
    return np.moveaxis(chw, 0, -1)


def _is_hdr(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext == ".exr" or ext == ".hdr"


def write_image(path: str, img: np.ndarray, gray: bool = False):
    hdr = _is_hdr(path)

    if not hdr:
        img = (img * 255).astype(np.uint8)
    if not gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img)


def read_image(path: str, gray: bool = False) -> np.ndarray:
    hdr = _is_hdr(path)
    # Read image
    if hdr:
        # print(path)
        img = pyexr.open(path).get()
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if not gray:  # Ensure correct color space
            # print(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if gray:  # Ensure single channel for gray scale
        img = ensureSingleChannel(img)

    if hdr:  # Always return float
        return img.astype(np.float32)
    else:
        return img.astype(np.float32) / 255

def read_mask(path: str, gray: bool = True) -> np.ndarray:
    mask = read_image(path, gray)
    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0
    mask = erosion(
        mask[..., 0], disk(3)
    )  # Apply a erosion (channels need to be removed)
    # mask = np.expand_dims(mask, -1) # And added back
    return mask


def ensureSingleChannel(x: np.ndarray) -> np.ndarray:
    ret = x
    if len(x.shape) > 2:
        ret = ret[:, :, 0]

    return np.expand_dims(ret, -1)


def compressDepth(
    d: np.ndarray, sigma: float = 2.5, epsilon: float = 0.7
) -> np.ndarray:
    """Compresses the full depth range to a 0 to 1 range. The possible depth range
    is modelled by sigma and epsilon and with sigma=2.5 and epsilon=0.7
    it is between 0.17 and 1.4.
    """
    d[d <= 0] = 1e-6
    return np.clip((-epsilon * d + 1) / (2 * sigma * d), 0, 1)


def uncompressDepth(
    d: np.ndarray, sigma: float = 2.5, epsilon: float = 0.7
) -> np.ndarray:
    """From 0-1 values to full depth range. The possible depth range
    is modelled by sigma and epsilon and with sigma=2.5 and epsilon=0.7
    it is between 0.17 and 1.4.
    """
    return 1 / (2 * sigma * d + epsilon)


def sRGBToLinear(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def linearTosRGB(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0.0031308, 1.055 * np.power(x, 1.0 / 2.4) - 0.055, x * 12.92)


def convert_luminance(x: np.ndarray) -> np.ndarray:
    return 0.212671 * x[:, :, 0] + 0.71516 * x[:, :, 1] + 0.072169 * x[:, :, 2]


def center_weight(x):
    def smoothStep(x, edge0=0.0, edge1=1.0):
        x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return x * x * x * (x * (x * 6 - 15) + 10)

    idx = np.argwhere(np.ones_like(x))
    idxs = np.reshape(idx, (x.shape[0], x.shape[1], 2))
    center_dist = np.linalg.norm(
        idxs - np.array([x.shape[0] / 2, x.shape[1] / 2], dtype=np.float32), axis=2
    )

    return 1 - smoothStep(center_dist / x.shape[1] * 2)


def compute_avg_luminance(x: np.ndarray) -> np.ndarray:
    L = convert_luminance(x)
    L = L * center_weight(L)
    avgL1 = np.average(L, axis=(0, 1))
    return avgL1


def compute_ev100_from_avg_luminance(avgL):
    return np.log2(avgL * 100.0 / 12.5)  # or 12.7


def convert_ev100_to_exp(ev100):
    maxL = 1.2 * np.power(2.0, ev100)
    return np.clip(1.0 / maxL, 1e-7, None)


def calculate_ev100_from_metadata(aperture_f: float, shutter_s: float, iso: int):
    ev_s = np.log2((aperture_f * aperture_f) / shutter_s)
    ev_100 = ev_s - np.log2(iso / 100)
    return ev_100


def compute_auto_exp(x: np.ndarray, clip: bool = True) -> np.ndarray:
    avgL = np.clip(compute_avg_luminance(x), 1e-5, None)
    ev100 = compute_ev100_from_avg_luminance(avgL)
    exp = convert_ev100_to_exp(ev100)  # This can become an invalid number. why?

    exposed = x * exp
    if clip:
        exposed = np.clip(exposed, 0.0, 1.0)

    return exposed, exp
