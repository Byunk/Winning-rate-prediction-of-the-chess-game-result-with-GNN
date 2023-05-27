import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from .encoder import Encoder
from .decoder import Decoder


class ChessModel(pl.LightningModule):
    def __init__(
        self,
        node_feature_dim,
        num_layers,
        heads,
        num_node,
        edge_index_doubled,
        edge_attr_doubled,
        learning_rate=1e-6,
    ):
        super().__init__()
        self.edge_index_doubled = edge_index_doubled.to(self.device)
        self.edge_attr_doubled = edge_attr_doubled.to(self.device)
        self.save_hyperparameters(
            "node_feature_dim", "num_layers", "heads", "learning_rate"
        )

        # init layers
        self.Encoder = Encoder(
            node_feature_dim,
            node_feature_dim,
            node_feature_dim,
            num_node,
            num_layers,
            heads,
        )
        self.Decoder = Decoder(2 * node_feature_dim, [100, 100], 3)

        # init parameters reccursively
        self.apply(self._init_parameters)

    def _init_parameters(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.LayerNorm):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    #### forward pass ###################################################
    def _shared_step(self, batch, edge_index_doubled, edge_attr_doubled):
        # 0. unpack batch
        batch_edge_index, batch_edge_attr = batch
        batch_edge_index = batch_edge_index.T

        edge_index_doubled = edge_index_doubled.to(self.device)
        edge_attr_doubled = edge_attr_doubled.to(self.device)

        # 1. Forward pass through Encoder
        node_features = self.Encoder(edge_index_doubled, edge_attr_doubled)

        # 2. Sample node features
        sampled_node_features = torch.cat(
            [node_features[batch_edge_index[0]], node_features[batch_edge_index[1]]],
            dim=-1,
        )

        # 3. Forward pass through Decoder
        outputs = self.Decoder(sampled_node_features)

        # 4. Set targets
        targets = batch_edge_attr

        loss = F.mse_loss(outputs, targets)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.edge_index_doubled, self.edge_attr_doubled)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.edge_index_doubled, self.edge_attr_doubled)
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        loss = self._shared_step(batch, self.edge_index_doubled, self.edge_attr_doubled)
        return loss

    #####################################################################

    #### configure optimizer
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, mode="min"),
                "monitor": "val_loss",
            },
        }
