import numpy as np
import torch
import trimesh

from src.data.path_handling import path_manager
from src.util.images import *


class TwoShotBrdfData(torch.utils.data.Dataset):
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
    
    def __init__(self, split):
        """
        :param split: one of 'train', 'val', 'test' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ["train", "val", "overfit", "test"]
        self.items = TwoShotBrdfData.items_subsets[split]
        self.prefix = TwoShotBrdfData.items_prefixes[split]

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of brdf data
        """
        item = self._gen_path(index)

        return {
            "cam1" :        readEXR(item / "cam1_env.exr")[0],
            "flash" :       readEXR(item / "cam1_flash.exr")[0],
            "cam2" :        readEXR(item / "cam2.exr")[0],
            "depth" :       readEXR(item / "depth.exr")[1],
            "diffuse" :     load_rgb(item / "diffuse.png"),
            "mask" :        load_mono(item / "mask.png"),
            "normal" :      readEXR(item / "normal.exr")[0],
            "roughness" :   load_mono(item / "roughness.png"),
            "specular" :    load_mono(item / "specular.png")
        }

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
        fdr_ = ((4 - int(fdr_idx_ / 10)) * "0") + str(fdr_idx_)
        itm_ = ((2 - int(itm_idx_ / 10)) * "0") + str(itm_idx_)
        return path_manager.data_dir / self.prefix / fdr_ / itm_

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        pass
