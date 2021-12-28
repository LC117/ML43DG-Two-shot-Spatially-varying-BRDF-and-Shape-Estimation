import numpy as np
import torch
import trimesh

from path_handling import *


class TwoShotBrdfData(torch.utils.data.Dataset):
    """
    Dataset for loading brdf data
    """

    def __init__(self, split):
        """
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of brdf data
                 "name", shape_identifier of the shape
        """
        return {
            "name" : "wip"
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO: Implement
        return 0

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        pass
