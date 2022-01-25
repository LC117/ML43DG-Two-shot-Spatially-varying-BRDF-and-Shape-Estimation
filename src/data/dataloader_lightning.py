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
                 persistent_workers: bool = False, pin_memory: bool = False):
        super().__init__()
        self.mode = mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        
        self.train =    "train" if not overfit else "overfit"
        self.val =      "val"   if not overfit else "overfit"
        self.test =     "test"  if not overfit else "overfit"

    def train_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.train, mode=self.mode), batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory, shuffle=True)

    def val_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.val, mode=self.mode), batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(TwoShotBrdfData(split=self.test, mode=self.mode), batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers, pin_memory=self.pin_memory)
        