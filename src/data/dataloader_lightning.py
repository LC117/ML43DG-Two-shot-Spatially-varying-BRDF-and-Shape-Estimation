import imp
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data._dataloader import TwoShotBrdfData

class TwoShotBrdfDataLightning(pl.LightningDataModule):
    """ Dummy extension to the PyTorch module to facilitate usage with Lightning's Trainer()
       
    Note:
        Overfit split implicitly included in Trainer() see 
        https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#overfit-batches 
        BUT as the overfit dataset is included in this repo it will be handled anyway.
        
    """
    def __init__(self, mode: str, batch_size: int = 64, num_workers=0, overfit: bool = False,
                 persistent_workers: bool = False, pin_memory: bool = False, shuffle: bool = True, use_gt: bool = False):
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.shuffle = (not overfit) and shuffle
        self.use_gt = use_gt
        
        self.train =    "train" if not overfit else "overfit"
        self.val =      "val"   if not overfit else "overfit"
        self.test =     "test"  if not overfit else "overfit"

    def train_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.train, training=True, mode=self.mode, use_gt=self.use_gt), batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.val, training=True, mode=self.mode, use_gt=self.use_gt),
                          batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.test, training=False, mode=self.mode, use_gt=self.use_gt),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def predict_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.val, training=False, mode=self.mode, use_gt=self.use_gt),
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)
        