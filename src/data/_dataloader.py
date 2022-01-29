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
        "train": (1, 99, 1000),
        "val": (0, 0, 1000),
        "test": (0, 0, 20),
        "overfit": (0, 0, 10)
    }

    items_prefixes = {
        "train": "CVPR20-TwoShotBRDFAndShapeDataset/training/",
        "val": "CVPR20-TwoShotBRDFAndShapeDataset/training/",
        "test": "CVPR20-TwoShotBRDFAndShapeDataset/testing/",
        "overfit": "CVPR20-TwoShotBRDFAndShapeDataset/overfit/"
    }

    def __init__(self, split, training, mode="joint", use_gt=False):
        """
        :param split: one of 'train', 'val', 'test' or 'overfit' - for training, validation or overfitting split
        :param training: bool -> Set to False for inference, to True for training!
        :param mode: one of 'inference', 'shape', 'illumination', 'svbrdf'/'joint' - We do not need to load all the data for training the first two networks
        :param gt: Ground truth, either use ground truth from the dataset 'theirs', or use predictions from previous passes 'ours'
        """
        super().__init__()

        assert split in ["train", "val", "overfit", "test"]
        assert mode in ["inference", "shape", "illumination", "svbrdf", "joint"]
        assert not (training and mode == "inference")  # Either one of these, not both

        self.items = TwoShotBrdfData.items_subsets[split]
        self.prefix = TwoShotBrdfData.items_prefixes[split]
        self.mode = mode
        self.split = split
        self.storeData = split == "overfit"
        self.data = {}
        self.use_gt = use_gt

        self.training = training  # Set to False for inference!

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
        
        if self.mode in ["inference", "shape", "illumination", "svbrdf", "joint"]:
            res.update({
                "cam1": self.read_and_transform(path_to_folder, ParameterNames.INPUT_1),
                "cam2": self.read_and_transform(path_to_folder, ParameterNames.INPUT_2),
                "mask": self.read_and_transform(path_to_folder, ParameterNames.MASK)
            })
        else:
            raise RuntimeError("Dataloader Mode not valid!")
        
        if not self.training:
            return res

        if self.mode == "shape":
            res.update({
                "depth": self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                "normal": self.read_and_transform(path_to_folder, ParameterNames.NORMAL)
            })
        elif self.mode == "illumination":
            if self.use_gt:
                res.update({
                    "depth": self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                    "normal": self.read_and_transform(path_to_folder, ParameterNames.NORMAL),
                    "sgs": self.read_and_transform(path_to_folder, ParameterNames.SGS)
                })
            else:
                res.update({
                    "depth_pred": self.read_and_transform(path_to_folder, ParameterNames.DEPTH_PRED),
                    "normal_pred": self.read_and_transform(path_to_folder, ParameterNames.NORMAL_PRED),
                    "sgs": self.read_and_transform(path_to_folder, ParameterNames.SGS)
                })
        elif self.mode == "svbrdf":
            if self.use_gt:
                res.update({
                    "depth": self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                    "normal": self.read_and_transform(path_to_folder, ParameterNames.NORMAL),
                    "sgs": self.read_and_transform(path_to_folder, ParameterNames.SGS),
                    "diffuse": self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE),
                    "specular": self.read_and_transform(path_to_folder, ParameterNames.SPECULAR),
                    "roughness": self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS)
                })
            else:
                res.update({
                    "depth": self.read_and_transform(path_to_folder, ParameterNames.DEPTH_PRED),
                    "normal": self.read_and_transform(path_to_folder, ParameterNames.NORMAL_PRED),
                    "sgs": self.read_and_transform(path_to_folder, ParameterNames.SGS_PRED),
                    "diffuse": self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE),
                    "specular": self.read_and_transform(path_to_folder, ParameterNames.SPECULAR),
                    "roughness": self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS)
                })
        elif self.mode == "joint":
            if self.use_gt:
                res.update({
                    "depth": self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                    "normal": self.read_and_transform(path_to_folder, ParameterNames.NORMAL),
                    "sgs": self.read_and_transform(path_to_folder, ParameterNames.SGS),
                    "diffuse": self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE),
                    "specular": self.read_and_transform(path_to_folder, ParameterNames.SPECULAR),
                    "roughness": self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS),
                    "diffuse_gt": self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE),
                    "specular_gt": self.read_and_transform(path_to_folder, ParameterNames.SPECULAR),
                    "roughness_gt": self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS),
                    "normal_gt": self.read_and_transform(path_to_folder, ParameterNames.NORMAL),
                    "depth_gt": self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                    "flash": self.read_and_transform(path_to_folder, ParameterNames.INPUT_1_FLASH)
                })
                res.update({
                    "rerender_img": res["cam1"]
                })
            else:
                res.update({
                    "depth": self.read_and_transform(path_to_folder, ParameterNames.DEPTH_PRED),
                    "normal": self.read_and_transform(path_to_folder, ParameterNames.NORMAL_PRED),
                    "sgs": self.read_and_transform(path_to_folder, ParameterNames.SGS_PRED),
                    "diffuse": self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE_PRED),
                    "specular": self.read_and_transform(path_to_folder, ParameterNames.SPECULAR_PRED),
                    "roughness": self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS_PRED),
                    "diffuse_gt": self.read_and_transform(path_to_folder, ParameterNames.DIFFUSE),
                    "specular_gt": self.read_and_transform(path_to_folder, ParameterNames.SPECULAR),
                    "roughness_gt": self.read_and_transform(path_to_folder, ParameterNames.ROUGHNESS),
                    "normal_gt": self.read_and_transform(path_to_folder, ParameterNames.NORMAL),
                    "depth_gt": self.read_and_transform(path_to_folder, ParameterNames.DEPTH),
                    "rerender_img": self.read_and_transform(path_to_folder, ParameterNames.RERENDER),
                    "flash": self.read_and_transform(path_to_folder, ParameterNames.INPUT_1_FLASH)
                })
        
        if self.storeData:
            self.data[index] = res

        return res

    def __len__(self):
        """
        :return: length of the dataset
        """
        s_idx_, e_idx_, n = self.items
        return (e_idx_ - s_idx_ + 1) * n

    def gen_path(self, index):
        return self._gen_path(index)

    def _gen_path(self, index):
        """
        A little bit of path hacking to transform index to filepath
        """
        s_idx_, e_idx_, n = self.items
        fdr_idx_ = s_idx_ + int(index / n)
        itm_idx_ = index % n

        # fill fdr_ with leading zeros, so that it always has 5 digits
        fdr_ = str(fdr_idx_).zfill(5)  # if not self.split == "overfit" else "00000"
        itm_ = str(itm_idx_).zfill(3)

        # fdr_ = ((4 - int(fdr_idx_ / 10)) * "0") + str(fdr_idx_)
        # itm_ = ((2 - int(itm_idx_ / 10)) * "0") + str(itm_idx_)
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
                # cam1_flash = pyexr.open(str(path_to_folder / ParameterNames.INPUT_1_FLASH.value)).get()
                # cam1_env = pyexr.open(str(path_to_folder / ParameterNames.INPUT_1_ENV.value)).get()
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
            #cam2 = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.INPUT_2.value)).get(), (0, 1, 2))#, (2, 0, 1))
            cam2 = read_image(str(path_to_folder / ParameterNames.INPUT_2.value), False)
            cam2 = TwoShotBrdfData.process_input_image(cam2)
            return np.transpose(cam2, (2, 0, 1))

        elif par_name == ParameterNames.MASK:  # DONE
            # mask = load_mono(path_to_folder / ParameterNames.MASK)[np.newaxis, ...]
            mask = read_image(str(path_to_folder / ParameterNames.MASK.value), True)
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            mask = erosion(
                mask[..., 0], disk(3)
            )  # Apply a erosion (channels need to be removed)
            # mask = np.expand_dims(mask, -1) # And added back
            return mask[np.newaxis, ...]

        elif par_name == ParameterNames.DEPTH:  # DONE
            # depth = pyexr.open(str(path_to_folder / ParameterNames.DEPTH)).get()[np.newaxis, :, :, 0]
            depth = read_image(str(path_to_folder / ParameterNames.DEPTH.value), True)
            depth = compressDepth(depth)
            return depth[np.newaxis, :, :, 0]

        elif par_name == ParameterNames.DEPTH_PRED:
            depth = read_image(str(path_to_folder / ParameterNames.DEPTH_PRED.value).replace("%d", "0"), True)
            depth = compressDepth(depth)
            return depth[np.newaxis, :, :, 0]

        elif par_name == ParameterNames.NORMAL:  # DONE
            # normal = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.NORMAL)).get(), (2, 0, 1))
            normal = read_image(str(path_to_folder / ParameterNames.NORMAL.value), False)
            return np.transpose(normal, (2, 0, 1))

        elif par_name == ParameterNames.NORMAL_PRED:  # DONE
            # normal = np.transpose(pyexr.open(str(path_to_folder / ParameterNames.NORMAL)).get(), (2, 0, 1))
            normal = read_image(str(path_to_folder / ParameterNames.NORMAL_PRED.value).replace("%d", "0"), False)
            return np.transpose(normal, (2, 0, 1))

        elif par_name == ParameterNames.SGS:  # DONE
            sgs = np.load(path_to_folder / ParameterNames.SGS.value).astype(np.float32)
            return sgs

        elif par_name == ParameterNames.DIFFUSE:  # DONE
            # diffuse = np.transpose(load_rgb(path_to_folder / "diffuse.png"), (2, 0, 1))
            diffuse = read_image(str(path_to_folder / ParameterNames.DIFFUSE.value), False)
            return np.transpose(diffuse, (2, 0, 1))

        elif par_name == ParameterNames.SPECULAR:  # DONE
            # specular = np.transpose(load_rgb(path_to_folder / "specular.png"), (2, 0, 1))
            specular = read_image(str(path_to_folder / ParameterNames.SPECULAR.value), False)
            return np.transpose(specular, (2, 0, 1))

        elif par_name == ParameterNames.ROUGHNESS:  # DONE
            # roughness = load_mono(path_to_folder / "roughness.png")[np.newaxis, :, :]
            roughness = read_image(str(path_to_folder / ParameterNames.ROUGHNESS.value), True)
            return roughness[np.newaxis, :, :, 0]
        
        elif par_name == ParameterNames.INPUT_1_FLASH:
            cam1_flash = read_image(str(path_to_folder / ParameterNames.INPUT_1_FLASH.value), False)
            cam1 = TwoShotBrdfData.process_input_image(cam1_flash)
            return np.transpose(cam1, (2, 0, 1))
        
        else:
            raise Exception("Parameter name not available!")

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        for key_ in batch:
            batch[key_].to(device)

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
