from pytorch_lightning.cli import LightningCLI
import sys
import numpy as np

# Modules
from module.ChessModel import ChessModel

# Data_Modules
from dataModule.MNIST import MNIST
from callback.file_lr import FileLRCallback

# Callbacks
# from callback.LogTestImageCallback import LogTestImageCallback


def cli_main():
    cli = LightningCLI(datamodule_class=MNIST)


if __name__ == "__main__":
    cli_main()
