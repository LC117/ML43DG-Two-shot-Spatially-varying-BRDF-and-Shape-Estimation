import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from model.illumination_model import IlluminationNetwork

def main():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(callbacks=[lr_monitor])

if __name__ == "__main__":
    main()

