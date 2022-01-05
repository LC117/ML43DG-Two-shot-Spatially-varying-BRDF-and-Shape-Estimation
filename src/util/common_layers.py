import torch
from torch import nn

def INReLU():
    """Shorthand for InstaceNorm + Relu

    Returns:
        [type]: [description]
    """
    return torch.nn.Sequential(
        nn.InstanceNorm2d(10, eps=1e-5, affine=True),
        nn.ReLU())