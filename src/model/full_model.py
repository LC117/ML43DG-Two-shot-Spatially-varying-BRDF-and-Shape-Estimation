import numpy as np
import cv2
import pyexr
import os
import pathlib
import torch

from src.model.shape_model import ShapeNetwork
from src.model.illumination_model import IlluminationNetwork
from src.model.svbrdf_model import SVBRDF_Network
from src.model.joint_model import JointNetwork

from src.utils.preprocessing_utils import read_image, read_mask
from src.utils.preprocessing_utils import sRGBToLinear

from src.utils.inference_renderer import Renderer, InferenceStage
from src.utils.visualize_tools import save



def full_inference(path_to_default_img, path_to_flash, path_to_mask):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    shape_net = ShapeNetwork(device=device).load_from_checkpoint(checkpoint_path=str(r"src/data/models/version_142/checkpoints/epoch=10-step=17016.ckpt"))
    ill_net = IlluminationNetwork().load_from_checkpoint(checkpoint_path=str(r"src/data/models/version_169/checkpoints/epoch=2-step=37124.ckpt"))
    brdf_net = SVBRDF_Network().load_from_checkpoint(checkpoint_path=str(r"src/data/models/version_222/checkpoints/epoch=3-step=66667.ckpt"))
    joint_net = JointNetwork()#.load_from_checkpoint(checkpoint_path=str(r"src/data/models/version_155/checkpoints/epoch=7-step=12375.ckpt"))
    
    location = pathlib.Path(path_to_default_img).parent
    assert str(location) == str(pathlib.Path(path_to_flash).parent)
    
    default_image =  sRGBToLinear(read_image(path_to_default_img)) # cam2
    flash_image =  sRGBToLinear(read_image(path_to_flash)) # cam1
    mask = read_mask(path_to_mask, gray=False)
    
    default_image = torch.tensor(default_image[np.newaxis, ...], device=device).permute(0, 3, 1, 2)
    flash_image = torch.tensor(flash_image[np.newaxis, ...], device=device).permute(0, 3, 1, 2)
    mask = torch.tensor(mask[np.newaxis, np.newaxis, :, :], device=device)
    
    # Pass the Shape Network:
    normal, depth = shape_net.forward((flash_image, default_image, mask))
    # the variable normal is called "res" in the original code
    # normal, depth = normal[0], depth[0]
    save(tensor_to_savable(normal), str(location / r"normal_pred0.exr"), also_as_png=True)
    save(tensor_to_savable(depth), str(location / r"depth_pred0.exr"), also_as_png=True)
    
    assert normal.max() <= 1. and depth.max() <= 1
    
    # Pass the Illumination Network
    sgs = ill_net.forward((flash_image, default_image, mask, normal, depth))
    save(sgs[0].detach().cpu().numpy(), str(location / r"sgs-pred.npy"))
    ill_net.render_and_store(sgs[0], path=str(location / r"sgs-pred.png"))
    
    # Pass the SVBRDF Network:
    diffuse, specular, roughness = brdf_net.forward((flash_image, default_image, mask, normal, depth))
    save(tensor_to_savable(diffuse), str(location / r"diffuse_pred0.exr"), also_as_png=True)
    save(tensor_to_savable(specular), str(location / r"specular_pred0.exr"), also_as_png=True)
    save(tensor_to_savable(roughness), str(location / r"roughness_pred0.exr"), also_as_png=True)
    # Rendering the SVBRDF Output:
    render = Renderer(InferenceStage.INITIAL_RENDERING).render_all(
        data=(diffuse, specular, roughness, normal, depth, mask, sgs))
    save(np.transpose(render.detach().cpu().numpy(), (1, 2, 0)), str(location / f"rerender0.exr"), also_as_png=True)
    
    # Pass the Joint Network:
    # Loss image etc. is calculated within the joint_net!
    diffuse, specular, roughness, normal, depth = joint_net.forward((flash_image, mask, normal, depth, sgs, render, roughness, diffuse, specular))
    save(tensor_to_savable(diffuse), str(location / r"diffuse_pred1.exr"), also_as_png=True)
    save(tensor_to_savable(specular), str(location / r"specular_pred1.exr"), also_as_png=True)
    save(tensor_to_savable(roughness), str(location / r"roughness_pred1.exr"), also_as_png=True)
    save(tensor_to_savable(normal), str(location / r"normal_pred1.exr"), also_as_png=True)
    save(tensor_to_savable(depth), str(location / r"depth_pred1.exr"), also_as_png=True)
    # Rendering the FINAL Output:
    render = Renderer(InferenceStage.INITIAL_RENDERING).render_all(
        data=(diffuse, specular, roughness, normal, depth, mask, sgs))
    save(np.transpose(render.detach().cpu().numpy(), (1, 2, 0)), str(location / f"rerender_final.exr"))
    
    assert diffuse.max() <= 1. and specular.max() <= 1 and roughness.max() <= 1
    
def tensor_to_savable(tensor, transpose=True):
    return np.transpose(tensor[0].detach().cpu().numpy(), (1, 2, 0))


if __name__ == "__main__":
    
    full_inference(path_to_default_img=r"inference_images/examples/0/cam2.png", 
                   path_to_flash=r"inference_images/examples/0/cam1.png", 
                   path_to_mask=r"inference_images/examples/0/mask.png"
                   )
    
    print("DONE!")