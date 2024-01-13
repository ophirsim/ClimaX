# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from extreme_weather_dataset import ExtremeWeatherDataset


class ExtremeWeatherDataModule(LightningDataModule):
    """DataModule for extreme weather data.

    Args:
        root_dir (str): Root directory for data.
        batch_size (int, optional): Batch size.
        num_workers (int, optional): Number of workers.
        pin_memory (bool, optional): Whether to pin memory.
    """

    def __init__(
        self,
        root_dir,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # Warn user about no support for several workers
        if num_workers > 1:
            raise NotImplementedError(
                "num_workers > 1 is not supported yet. Performance will likely degrage too with larger num_workers."
            )

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # construct the paths for the train, val, and test directories
        self.lister_train = os.path.join(root_dir, "train")
        self.lister_val = os.path.join(root_dir, "val")
        self.lister_test = os.path.join(root_dir, "test")

        # create the train, validation, and test attributes that will contain the dataloaders
        self.data_train: Optional[ExtremeWeatherDataset] = None
        self.data_val: Optional[ExtremeWeatherDataset] = None
        self.data_test: Optional[ExtremeWeatherDataset] = None

    def setup(self):
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = ExtremeWeatherDataset(self.lister_train, transform=None, target_transform=None)
            self.data_val = ExtremeWeatherDataset(self.lister_val, transform=None, target_transform=None)
            self.data_test = ExtremeWeatherDataset(self.lister_test, transform=None, target_transform=None)

    def train_dataloader(self):
        # return the dataloader that contains the training data
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        # return the dataloader that contains the validation data
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        # return the dataloader that contains the test data
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
