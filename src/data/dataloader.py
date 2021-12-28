import numpy as np
import torch
import trimesh

from src.data.path_handling import path_manager


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
                 "name", shape_identifier of the shape
        """
        return {
            "name" : "foo"
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO: Implement
        s_idx_, e_idx_, n = self.items
        return (e_idx_ - s_idx_ + 1) * n

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        pass
