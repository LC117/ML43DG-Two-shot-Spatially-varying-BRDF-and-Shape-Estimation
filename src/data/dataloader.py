import numpy as np
import torch
import trimesh

from torch.utils.data import Dataset
from src.data.path_handling import path_manager
from src.utils.images import *

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
    
    def __init__(self, split, mode="all"):
        """
        :param split: one of 'train', 'val', 'test' or 'overfit' - for training, validation or overfitting split
        :param mode: one of 'cams', 'shape', 'all' - We do not need to load all the data for training the first two networks
        """
        super().__init__()
        assert split in ["train", "val", "overfit", "test"]
        assert mode in ["cams", "shape", "illumination", "all"]
        self.items = TwoShotBrdfData.items_subsets[split]
        self.prefix = TwoShotBrdfData.items_prefixes[split]
        self.mode = mode

        self.storeData = split == "overfit"
        self.data = {}

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of brdf data
        """
        if self.storeData:
            if index in self.data:
                return self.data[index]

        item = self._gen_path(index)
        res = {}
        if self.mode in ["cams", "shape", "illumination", "all"]:
            res = {
                "cam1" :        pyexr.open(str(item / "cam1_env.exr")).get(),
                "cam2" :        pyexr.open(str(item / "cam2.exr")).get(),
                "mask" :        load_mono(item / "mask.png")
            }
        if self.mode in ["shape", "illumination", "all"]:
            res.update({
                "depth" :       pyexr.open(str(item / "depth.exr")).get()[:, :, 0],
                "normal" :      pyexr.open(str(item / "normal.exr")).get(),
            })
        if self.mode in ["illumination", "all"]:
            res.update({
                "sgs" :         np.load(item / "sgs.npy").astype(np.float32)
            })
        if self.mode == "all":
            res.update({
                "flash" :       pyexr.open(str(item / "cam1_flash.exr")).get(),
                "diffuse" :     load_rgb(item / "diffuse.png"),
                "specular" :    load_rgb(item / "specular.png"),
                "roughness" :   load_mono(item / "roughness.png")
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
        fdr_ = str(fdr_idx_).zfill(5)
        itm_ = str(itm_idx_).zfill(3)

        #fdr_ = ((4 - int(fdr_idx_ / 10)) * "0") + str(fdr_idx_)
        #itm_ = ((2 - int(itm_idx_ / 10)) * "0") + str(itm_idx_)
        return path_manager.data_dir / self.prefix / fdr_ / itm_

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

if __name__ == "__main__":
    data = TwoShotBrdfData(split="overfit", mode="shape")
    print("test")