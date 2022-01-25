import numpy as np
import torch
import trimesh

from torch.utils.data import Dataset
from typing import Tuple
from skimage.morphology import disk, erosion


from src.data.path_handling import path_manager
from src.utils.images import *
from src.utils.config import ParameterNames
from src.utils.preprocessing_utils import read_image, compressDepth, compute_auto_exp

import pyexr

class TwoShotBrdfData(Dataset):
    """
    Dataset for loading brdf data
    """

    # Split dataset into four subsets
    #   [0]: filesystem start index
    #   [1]: filesystem end index
    #   [2]: items per folder
    items_subsets = {
        "train" :       (1, 99, 1000),
        "val" :         (0, 0, 1000),
        "test" :        (0, 0, 20),
        "overfit" :     (0, 0, 10)
    }

    items_prefixes = {
        "train" :       "CVPR20TwoShotBRDFAndShapeDataset/training/",
        "val" :         "CVPR20TwoShotBRDFAndShapeDataset/training/",
        "test" :        "CVPR20TwoShotBRDFAndShapeDataset/testing/",
        "overfit" :     "CVPR20TwoShotBRDFAndShapeDataset/overfit/"
    }
    
    def __init__(self, split, training, mode="all"):
        """
        :param training: bool -> Set to False for inference, to True for training!
        :param split: one of 'train', 'val', 'test' or 'overfit' - for training, validation or overfitting split
        :param mode: one of 'cams', 'shape', 'all' - We do not need to load all the data for training the first two networks
        """
        super().__init__()
        
        assert split in ["train", "val", "overfit", "test"]
        assert mode in ["inference", "shape", "illumination", "svbrdf", "joined"]
        assert not (training and mode == "inference") # Either one of these, not both
        
        self.items = TwoShotBrdfData.items_subsets[split]
        self.prefix = TwoShotBrdfData.items_prefixes[split]
        self.mode = mode
        self.split = split
        self.storeData = split == "overfit"
        self.data = {}
        
        self.training = training # Set to False for inference!

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of brdf data
        
        Note: 
        Tansformation of shape is done here, as in official Datasets from torchvision.
        see: https://pytorch.org/vision/stable/_modules/torchvision/datasets/mnist.html#MNIST
        """
        if self.storeData:
            if index in self.data:
                return self.data[index]
        

        path_to_folder = self._gen_path(index)
        res = {}
        if self.mode in ["shape", "illumination", "svbrdf_or_joined"]:
            res.update({
                "cam1" :        self.read_and_transform(path_to_folder, ParameterNames.INPUT_1),
                "cam2" :        self.read_and_transform(path_to_folder, ParameterNames.INPUT_2),
                "mask" :        self.read_and_transform(path_to_folder, ParameterNames.MASK)
            })
            if not self.training:
                return res
            res.update({
                "depth" :       self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                "normal" :      self.read_and_transform(path_to_folder, ParameterNames.NORMAL)
            })
            
        if self.mode in ["illumination", "svbrdf_or_joined"]:
            res.update({
                "sgs" :         self.read_and_transform(path_to_folder, ParameterNames.SGS)
            })
            
        if self.mode == "svbrdf_or_joined":
            res.update({
                # "flash" :       np.transpose(pyexr.open(str(item / "cam1_flash.exr")).get(), (2, 0, 1)), # SHOULD NOT BE USED -> cam1_env and cam1_flash need to be merged! -> use cam1
                "diffuse" :     self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE),
                "specular" :    self.read_and_transform(path_to_folder, ParameterNames.SPECULAR),
                "roughness" :   self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS)
            })
            
        if self.storeData:
            self.data[index] = res
            
        return res

    """
    def __getitem__(self, index):
        #
        #PyTorch requires you to provide a getitem implementation for your dataset.
        #:param index: index of the dataset sample that will be returned
        #:return: a dictionary of brdf data
        
        item = self._gen_path(index)
        res = {}
        if self.mode in ["cams", "shape", "illumination", "all"]:
            res = {
                "cam1" :        readEXR(item / "cam1_env.exr")[0],
                "cam2" :        readEXR(item / "cam2.exr")[0],
                "mask" :        load_mono(item / "mask.png")
            }
        if self.mode in ["shape", "illumination", "all"]:
            res.update({
                "depth" :       readEXR(item / "depth.exr")[1],
                "normal" :      readEXR(item / "normal.exr")[0],
            })
        if self.mode in ["illumination", "all"]:
            res.update({
                "sgs" :         np.load(item / "sgs.npy").astype(np.float32)
            })
        if self.mode == "all":
            res.update({
                "flash" :       readEXR(item / "cam1_flash.exr")[0],
                "diffuse" :     load_rgb(item / "diffuse.png"),
                "specular" :    load_rgb(item / "specular.png"),
                "roughness" :   load_mono(item / "roughness.png")
            })
        return res
    """

    def __len__(self):
        """
        :return: length of the dataset
        """
        s_idx_, e_idx_, n = self.items
        return (e_idx_ - s_idx_ + 1) * n

    def _gen_path(self, index):
        """
        A little bit of path hacking to transform index to filepath
        """
        s_idx_, e_idx_, n = self.items
        fdr_idx_ = s_idx_ + int(index / n)
        itm_idx_ = index % n

        # fill fdr_ with leading zeros, so that it always has 5 digits
        fdr_ = str(fdr_idx_).zfill(5) # if not self.split == "overfit" else "00000"
        itm_ = str(itm_idx_).zfill(3)

        #fdr_ = ((4 - int(fdr_idx_ / 10)) * "0") + str(fdr_idx_)
        #itm_ = ((2 - int(itm_idx_ / 10)) * "0") + str(itm_idx_)
        return path_manager.data_dir / self.prefix / fdr_ / itm_

        
    def read_and_transform(self, path_to_folder, par_name: ParameterNames):
        """
        Function to read and apply transformations to the cam images according to the preprocessing:
        see: https://github.com/NVlabs/two-shot-brdf-shape/blob/352201b66bfa5cd5e25111451a6583a3e7d499f0/dataflow/dataflow.py#L75
        
        Note:
            If self.training is False inputs are assumed to be of low dynamic range i.e. PNG Format!
        """
        if par_name == ParameterNames.INPUT_1:
            if self.training:
                # cam1_flash = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.INPUT_1_FLASH)).get(), (2, 0, 1))
                # cam1_env = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.INPUT_1_ENV)).get(), (2, 0, 1))
                cam1_flash = read_image(str(path_to_folder / ParameterNames.INPUT_1_FLASH.value), False)
                cam1_env = read_image(str(path_to_folder / ParameterNames.INPUT_1_ENV.value), False)
                cam1 = TwoShotBrdfData.merge_seperate_input_images(cam1_flash, input_env=cam1_env)
                cam1 = TwoShotBrdfData.process_input_image(cam1)
                
            else:
                # cam1 = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.INPUT_1_LDR)).get(), (2, 0, 1))
                cam1 = read_image(str(path_to_folder / ParameterNames.INPUT_1_LDR.value), False)
                cam1 = TwoShotBrdfData.process_input_image(cam1)
            return np.transpose(cam1, (2, 0, 1))
                
        elif par_name == ParameterNames.INPUT_2:
            # cam2 = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.INPUT_2)).get(), (2, 0, 1))
            cam2 = read_image(str(path_to_folder / ParameterNames.INPUT_2.value), False)
            cam2 = TwoShotBrdfData.process_input_image(cam2)
            return np.transpose(cam2, (2, 0, 1))
        
        elif par_name == ParameterNames.MASK: # DONE
            # mask = load_mono(path_to_folder / ParameterNames.MASK)[np.newaxis, ...]
            mask = read_image(str(path_to_folder / ParameterNames.MASK.value), True)
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            mask = erosion(
                mask[..., 0], disk(3)
            )  # Apply a erosion (channels need to be removed)
            # mask = np.expand_dims(mask, -1) # And added back
            return mask[np.newaxis, ...]
            
        elif par_name == ParameterNames.DEPTH: # DONE
            # depth = pyexr.open(str(path_to_folder / ParameterNames.DEPTH)).get()[np.newaxis, :, :, 0]
            depth = read_image(str(path_to_folder / ParameterNames.DEPTH.value), True)
            depth = compressDepth(depth)
            return depth[np.newaxis, :, :, 0]
        
        elif par_name == ParameterNames.NORMAL: # DONE
            # normal = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.NORMAL)).get(), (2, 0, 1))
            normal = read_image(str(path_to_folder / ParameterNames.NORMAL.value), False)
            return np.transpose(normal, (2, 0, 1))  
            
        elif par_name == ParameterNames.SGS: # DONE
            sgs = np.load(path_to_folder / ParameterNames.SGS.value).astype(np.float32)
            return sgs
            
        elif par_name == ParameterNames.DIFFUSE: # DONE
            # diffuse = np.transpose(load_rgb(path_to_folder / "diffuse.png"), (2, 0, 1))
            diffuse = read_image(str(path_to_folder / ParameterNames.DIFFUSE.value), False)
            return np.transpose(diffuse, (2, 0, 1))  
            
        elif par_name == ParameterNames.SPECULAR: # DONE
            # specular = np.transpose(load_rgb(path_to_folder / "specular.png"), (2, 0, 1))
            specular = read_image(str(path_to_folder / ParameterNames.SPECULAR.value), False)
            return np.transpose(specular, (2, 0, 1))  
        
        elif par_name == ParameterNames.ROUGHNESS: # DONE
            # roughness = load_mono(path_to_folder / "roughness.png")[np.newaxis, :, :]
            roughness = read_image(str(path_to_folder / ParameterNames.ROUGHNESS.value), False)
            return roughness[np.newaxis, :, :]
        else:
            raise Exception("Parameter name not available!")
        
    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch["cam1"].to(device)
        batch["cam2"].to(device)
        batch["mask"].to(device)
        if "depth" in batch.keys():
            batch["depth"].to(device)
            batch["normal"].to(device)
        if "sgs" in batch.keys():
            batch["sgs"].to(device)
        if "flash" in batch.keys():
            batch["flash"].to(device)
            batch["diffuse"].to(device)
            batch["roughness"].to(device)
            batch["specular"].to(device)
            
    @staticmethod
    def process_input_image(img: np.ndarray):
        img, _ = compute_auto_exp(img)
        return np.nan_to_num(img)

    @staticmethod
    def process_input_images(
        input1: np.ndarray, input2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        return (
            TwoShotBrdfData.process_input_image(input1),
            TwoShotBrdfData.process_input_image(input2),
        )

    @staticmethod
    def merge_seperate_input_images(
        input_flash: np.ndarray, input_env: np.ndarray, flash_strength: float = 1.0
    ) -> np.ndarray:
        if input_flash.shape[-1] > 3:
            flashMask = input_flash[:, :, 3:4]
            input_flash = input_flash[:, :, :3] * flashMask

        return input_env + input_flash * flash_strength
            


if __name__ == "__main__":
    data = TwoShotBrdfData(split="overfit", training=True, mode="shape")
    print("test")