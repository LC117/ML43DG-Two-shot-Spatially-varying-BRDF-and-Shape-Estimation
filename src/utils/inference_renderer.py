import numpy as np
import os
import time 
import torch
import pathlib

import src.utils.sg_utils as sg

from src.utils.rendering_layer import RenderingLayer
from src.config import get_render_config
from src.utils.common_layers import apply_mask
from enum import Enum, auto
from src.utils.visualize_tools import save

class InferenceStage(Enum):
    SHAPE = auto()
    ILLUMINATION = auto()
    BRDF = auto()
    INITIAL_RENDERING = auto()
    JOINT = auto()
    FINAL_RENDERING = auto()
    

class Renderer:
    def __init__(self, step: InferenceStage):
        self.render_config = get_render_config()
        assert (
            step == InferenceStage.INITIAL_RENDERING
            or step == InferenceStage.FINAL_RENDERING
        )
        self.step = step
        self.stage = (
            0
            if step == InferenceStage.INITIAL_RENDERING
            else 1
        )

    def render_all(self, data):
        t0 = time.time()

        render = self.render(data)

        t1 = time.time()
        print("Rendering finished in: {}".format(t1 - t0))
        return render

    def render(self, dp):
        # Batch everything
        # Extract corresponding maps
        diffuse, specular, roughness, normal, depth, mask, sgs = dp

        # Setup everything for rendering
        imgSize = diffuse.shape[2]
        num_sgs = sgs.shape[1]  # Get the number of sgs
        intensity = self.render_config["light_intensity_lumen"] / (4.0 * np.pi)
        light_color = np.asarray(self.render_config["light_color"])
        light_color_intensity = light_color * intensity

        camera_pos = torch.Tensor(self.render_config["camera_position"])[None, ...]
        
        light_pos = torch.Tensor(self.render_config["light_position"])[None, ...]

        axis_sharpness = torch.Tensor(sg.setup_axis_sharpness(num_sgs))[None, ...]
        
        # Add a batch dim
        light_col = torch.tensor(
            light_color_intensity.reshape([1, 3]), dtype=torch.float32
        )

        sgs_joined = torch.concat([sgs, axis_sharpness], -1)

        renderer = RenderingLayer(
            self.render_config["field_of_view"],
            self.render_config["distance_to_zero"],
            torch.Size([-1, 3, imgSize, imgSize]),
        )
        rerendered = renderer.call(
            diffuse,
            specular,
            roughness,
            normal,
            depth,
            mask,
            camera_pos,
            light_pos,
            light_col,
            sgs_joined,
        )
        rerendered = apply_mask(rerendered, mask)
        result = rerendered[0]
        return result
    
def tensor_to_savable(tensor, transpose=True):
    return np.transpose(tensor.detach().numpy(), (1, 2, 0))
    
if __name__ == "__main__":
    from src.data._dataloader import TwoShotBrdfData
    dataset = TwoShotBrdfData(split= "overfit", training=True, mode="joint", use_gt=True)
    data = dataset[0]
    location = pathlib.Path(r"inference_images/")
    diffuse, specular, roughness, normal, depth, mask, sgs = (
        data["diffuse"], data["specular"], data["roughness"], data["normal"], data["depth"], data["mask"], data["sgs"])
    
    render = Renderer(InferenceStage.INITIAL_RENDERING).render_all(data=tuple(torch.Tensor(d)[None, ...] for d in [diffuse, specular, roughness, normal, depth, mask, sgs]))
    save(tensor_to_savable(render), str(location / f"rerender0.exr"))
    print()