import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import random_split


class GraphDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir,
        proportion: list = [0.8, 0.1, 0.1],
        batch_size: int = 32,
        num_workers: int = 32,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_portion, self.val_portion, self.test_portion = proportion

    def setup(self, stage=None):
        data = torch.load(self.data_dir)
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]

        # size
        dataset_size = len(edge_attr)
        train_size = int(self.train_portion * dataset_size)
        val_size = int(self.val_portion * dataset_size)
        test_size = dataset_size - train_size - val_size
        print(dataset_size, train_size, val_size)

        # preprocessing
        epsilon = 1e-8
        edge_attr = edge_attr / (edge_attr.sum(dim=1, keepdim=True) + epsilon)
        edge_index = edge_index.T

        # random split
        self.data = TensorDataset(edge_index, edge_attr)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.data, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
