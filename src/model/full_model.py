import numpy as np
import cv2
import pyexr
import os
import pathlib
import torch

from src.model.shape_model import ShapeNetwork
from src.model.illumination_model import IlluminationNetwork
from src.model.svbrdf_model import SVBRDF_Network

from src.utils.preprocessing_utils import read_image, read_mask
from src.utils.preprocessing_utils import sRGBToLinear

from src.utils.inference_renderer import Renderer, InferenceStage




def full_inference(path_to_default_img, path_to_flash, path_to_mask):
    
    location = pathlib.Path(path_to_default_img).parent
    assert str(location) == str(pathlib.Path(path_to_flash).parent)
    
    default_image =  sRGBToLinear(read_image(path_to_default_img)) # cam2
    flash_image =  sRGBToLinear(read_image(path_to_flash)) # cam1
    mask = read_mask(path_to_mask, gray=False)
    
    default_image = torch.Tensor(default_image[np.newaxis, ...]).permute(0, 3, 1, 2)
    flash_image = torch.Tensor(flash_image[np.newaxis, ...]).permute(0, 3, 1, 2)
    mask = torch.Tensor(mask[np.newaxis, np.newaxis, :, :])
    
    # Pass the Shape Network:
    normal, depth = ShapeNetwork().forward((flash_image, default_image, mask))
    # the variable normal is called "res" in the original code
    # normal, depth = normal[0], depth[0]
    save(np.transpose(normal[0].detach().numpy(), (1, 2, 0)), str(location / r"normal_pred0.exr"), also_as_png=True)
    save(np.transpose(depth[0].detach().numpy(), (1, 2, 0)), str(location / r"depth_pred0.exr"), also_as_png=True)
    
    assert normal.max() <= 1. and depth.max() <= 1
    
    # Pass the Illumination Network
    sgs = IlluminationNetwork().forward((flash_image, default_image, mask, normal, depth))
    save(sgs, str(location / r"sgs-pred.npy"))
    
    # Pass the SVBRDF Network:
    diffuse, specular, roughness = SVBRDF_Network().forward(flash_image, default_image, mask, normal, depth)
    save(diffuse, str(location / r"diffuse_pred0.exr"))
    save(specular, str(location / r"specular_pred0.exr"))
    save(roughness, str(location / r"roughness_pred0.exr"))
    # Rendering the SVBRDF Output:
    render = Renderer(InferenceStage.INITIAL_RENDERING).render(data=[(diffuse, specular, roughness, normal, depth, mask, sgs)])
    save(render, str(location / f"rerender0.exr"))
    
    assert diffuse.max() <= 1. and specular.max() <= 1 and roughness.max() <= 1
    
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
            cv2.imwrite(save_path.split(".")[0] + ".png", data * 255)
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

if __name__ == "__main__":
    
    full_inference(path_to_default_img=r"inference_images/examples/0/cam2.png", 
                   path_to_flash=r"inference_images/examples/0/cam1.png", 
                   path_to_mask=r"inference_images/examples/0/mask.png"
                   )
    
    print("DONE!")