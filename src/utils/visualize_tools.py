import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch
import numpy as np


def save_img(img, path, name):
    """
    Save image to path
    """
    # for path_part in path.split('/'):
    if not os.path.exists(path):
        os.makedirs(path)

    if type(img) == torch.Tensor:
        img = img.detach().cpu().numpy()

    # remove from batch
    if len(img.shape) == 4:
        img = img[0]

    # make sure img is channel last
    if img.shape[0] == 1 or img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))

    # unsqueeze single channel imgs
    if img.shape[2] == 1:
        img = np.squeeze(img, axis=2)
        plt.imsave(path + name + ".png", img, cmap="gray", vmin=0, vmax=1)
    else:
        plt.imsave(path + name + ".png", img, vmin=0, vmax=1)


"""
# debug the consistency loss
depth_tensor = torch.Tensor(depth_gt)
near = uncompressDepth(1)
far = uncompressDepth(0)
d = uncompressDepth(depth_tensor)
depth_tensor = div_no_nan(d - near, far - near)

sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).detach()
sobel_x = sobel_x.view((1, 1, 3, 3))
sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).detach()
sobel_y = -1. * sobel_y.view((1, 1, 3, 3))

dx = torch.nn.functional.conv2d(depth_tensor.view((1, 1, 256, 256)), sobel_x, padding=1, stride=1)
dy = torch.nn.functional.conv2d(depth_tensor.view((1, 1, 256, 256)), sobel_y, padding=1, stride=1)

texel_size = 1.0 / 256
# create a tensor of ones of shape and type of out[..., 0]
ones = torch.ones_like(dx)
dz = ones * texel_size * 2.0

# n = tf.concat([dx, dy, dz], -1)
n = torch.cat([dx, dy, dz], dim=1)
# n = normalize(n)
n = n / torch.norm(n, dim=1, keepdim=True)
n = n * 0.5 + 0.5

n = n.squeeze(0)
n = n.permute(1, 2, 0)
dx = dx.squeeze(0).squeeze(0)
dy = dy.squeeze(0).squeeze(0)

# save the n as rgb using matplotlib
# plt.imsave("Test_Results/n.png", n.detach().cpu().numpy())

# save dx and dy using matplotlib
# plt.imsave("Test_Results/dx.png", dx.detach().cpu().numpy(), cmap="gray")
# plt.imsave("Test_Results/dy.png", dy.detach().cpu().numpy(), cmap="gray")

# save depth_tensor using matplotlib
# plt.imsave("Test_Results/depth_tensor.png", depth_tensor.detach().cpu().numpy(), cmap="gray")
"""