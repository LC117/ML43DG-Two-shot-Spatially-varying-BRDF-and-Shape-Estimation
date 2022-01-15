# https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/utils/sg_utils.py

import numpy as np
import os
from typing import List, Union

# import cv2
import numpy as np
# import pyexr
from tqdm import tqdm


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-12))


def dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * y, axis=-1, keepdims=True)


def normalize(x: np.ndarray) -> np.ndarray:
    return x / magnitude(x)

def setup_axis_sharpness(num_sgs) -> np.ndarray:
    axis = []
    inc = np.pi * (3.0 - np.sqrt(5.0))
    off = 2.0 / num_sgs
    for k in range(num_sgs):
        y = k * off - 1.0 + (off / 2.0)
        r = np.sqrt(1.0 - y * y)
        phi = k * inc
        axis.append(normalize(np.array([np.cos(phi) * r, np.sin(phi) * r, y])))

    minDp = 1.0
    for a in axis:
        h = normalize(a + axis[0])
        minDp = min(minDp, dot(h, axis[0]))

    sharpness = (np.log(0.65) * num_sgs) / (minDp - 1.0)

    axis = np.stack(axis, 0)  # Shape: num_sgs, 3
    sharpnessNp = np.ones((num_sgs, 1)) * sharpness
    return np.concatenate([axis, sharpnessNp], -1)