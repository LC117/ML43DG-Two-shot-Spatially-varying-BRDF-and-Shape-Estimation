import matplotlib.pyplot as plt
from pathlib import Path
import os
import torch
import numpy as np
from src.utils.preprocessing_utils import save
import pyexr
import cv2
from src.utils.preprocessing_utils import read_image, compressDepth, compute_auto_exp, read_mask


def save_img(img, path, name, use_plt=False, as_exr=False, normalized=True):
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

    file_type = '.exr' if as_exr else '.png'
    file_name = str(path) + name + file_type
    file_name = file_name.replace('\\', '/')

    # unsqueeze single channel imgs
    if img.shape[2] == 1:
        img = np.squeeze(img, axis=2)
        if use_plt:
            plt.imsave(file_name, img, cmap="gray", vmin=0, vmax=1)
            #if normalized:
            #    plt.imsave(file_name, img, cmap="gray", vmin=0, vmax=1)
            #else:
            #    plt.imsave(file_name, img, cmap="gray")
                #save(img, file_name, grayscale=True, alpha=False)
        else:
            save(img, file_name, grayscale=True, alpha=False)
    else:
        if use_plt:
            plt.imsave(file_name, img, vmin=0, vmax=1)
            #if normalized:
            #    plt.imsave(file_name, img, vmin=0, vmax=1)
            #else:
            #    plt.imsave(file_name, img)
                #save(img, file_name, grayscale=False, alpha=False)
        else:
            save(img, file_name, grayscale=False, alpha=False)

def _is_hdr(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext == ".exr" or ext == ".hdr"  
    
def save(
    data: np.ndarray, save_path: str, grayscale: bool = False, alpha: bool = False, also_as_png: bool=True
):
    """Saves the data to a specified path and handles all required extensions
    Args:
        img: The numpy RGB or Grayscale float image with range 0 to 1.
        save_path: The path the image is saved to.
        grayscale: True if the image is in grayscale, False if RGB.
        alpha: True if the image contains transparency, False if opaque 
    """
    hdr = _is_hdr(save_path)
    npy = os.path.splitext(save_path)[1] == ".npy"
    if hdr:
        pyexr.write(save_path, data)
        if also_as_png:
            data = cv2.cvtColor(data * 255, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path.split(".")[0] + ".png", data)
    elif npy:
        np.save(save_path, data)
    else:
        asUint8 = (data * 255).astype(np.uint8)
        if alpha:
            if grayscale:
                print("ALPHA AND GRAYSCALE IS NOT FULLY SUPPORTED")
            proc = cv2.COLOR_RGBA2BGRA
        elif not alpha and grayscale:
            proc = cv2.COLOR_GRAY2BGR
        else:
            proc = cv2.COLOR_RGB2BGR

        toSave = cv2.cvtColor(asUint8, proc)

        cv2.imwrite(save_path, toSave)
        if also_as_png and len(save_path.split(".png")) == 1:
            cv2.imwrite(save_path.split(".")[0]+".png", toSave)
            
            
            
            
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


def exr_to_jpg(path):
    img = read_image(path, False)
    save(img, path.replace(".exr", ".png"))
    
if __name__ == "__main__":
    exr_to_jpg(r"src/data/CVPR20-TwoShotBRDFAndShapeDataset/overfit/00000/000/cam2.exr")
    exr_to_jpg(r"src/data/CVPR20-TwoShotBRDFAndShapeDataset/overfit/00000/000/cam1_env.exr")
    exr_to_jpg(r"src/data/CVPR20-TwoShotBRDFAndShapeDataset/overfit/00000/000/cam1_flash.exr")
    print("")