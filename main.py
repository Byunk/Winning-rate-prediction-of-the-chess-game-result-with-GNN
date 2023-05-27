import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pandas as pd

from DataModule import GraphDataModule, ChessData
from Model import ELO, ChessModel

seed_everything(1024, workers=True)

GRAPH_DATA_DIR = "./data/graph_data.pt"
LOG_DIR = "logs"
SOURCE_NAME = "lichess_db_standard_rated_2014-12.pgn"


class ChessPredictor:
    def __init__(self):
        pass

    def preprocess(self, dir=SOURCE_NAME):
        print("Start preprocess...")
        self.chessData = ChessData(dir=dir)
        self.chessData.prepare()
        data = torch.load(self.chessData.pt_dir)
        edge_index = data["edge_index"]
        edge_attr = data["edge_attr"]
        self.num_node = data["num_node"]

        # Inflate edge attr
        self.edge_index_doubled = edge_index.repeat(1, 2)
        self.edge_attr_doubled = self.inflate_edge_attr(edge_attr)

        graphData = GraphDataModule(
            data_dir=self.chessData.pt_dir, batch_size=32, num_workers=32
        )
        graphData.prepare_data()
        graphData.setup(stage="fit")
        self.train_dataloader = graphData.train_dataloader()
        self.val_dataloader = graphData.val_dataloader()
        self.test_dataloader = graphData.test_dataloader()

        self.test_edge_index = graphData.test_dataset.dataset.tensors[0][
            graphData.test_dataset.indices
        ]
        self.test_edge_attr = graphData.test_dataset.dataset.tensors[1][
            graphData.test_dataset.indices
        ]

        print("End preprocess...")

    def inflate_edge_attr(self, edge_attr):
        ones = torch.ones_like(edge_attr[:, [0]])
        zeros = torch.zeros_like(edge_attr[:, [0]])
        edge_attr_WB = torch.concat([edge_attr, zeros], dim=1)
        edge_attr_BW = torch.concat([torch.flip(edge_attr, dims=[1]), ones], dim=1)
        return torch.concat([edge_attr_WB, edge_attr_BW], dim=0)

    def train_gnn(self):
        logger = TensorBoardLogger(LOG_DIR, name="node_feature_dim:32")
        callbacks = [
            ModelCheckpoint(monitor="val_loss", mode="min"),
            EarlyStopping(monitor="val_loss", mode="min", patience=3),
        ]
        self.trainer = Trainer(
            logger=logger, callbacks=callbacks, val_check_interval=0.1, max_epochs=1000
        )

        self.chessModel = ChessModel(
            node_feature_dim=32,
            num_layers=2,
            heads=3,
            num_node=self.num_node,
            edge_index_doubled=self.edge_index_doubled,
            edge_attr_doubled=self.edge_attr_doubled,
        )

        self.trainer.fit(self.chessModel, self.train_dataloader, self.val_dataloader)

    def predict_gnn(self):
        self.trainer.predict(self.chessModel, self.test_dataloader)

    def train_elo(self):
        test_edge_index = pd.DataFrame(
            self.test_edge_index.numpy(), columns=["white", "black"]
        )
        self.elo = ELO(self.chessData.csv_dir, test_edge_index)
        self.elo.train()

    def predict_elo(self):
        outputs = self.elo.predict()
        loss = F.mse_loss(outputs, self.test_edge_attr)
        print(loss)


def main():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from textwrap import dedent

    parser = ArgumentParser(
        formatter_class=RawDescriptionHelpFormatter,
        description=dedent(
            """\
            Script to predict chess game result based on GNN or ELO
            """
        ),
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Method: gnn or elo",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="source directory",
    )
    args = parser.parse_args()

    if args.dir != None:
        chess = ChessPredictor(args.dir)
    else:
        chess = ChessPredictor()
    chess.preprocess()

    if args.method == "gnn":
        chess.train_gnn()
        chess.predict_gnn()
    elif args.method == "elo":
        chess.train_elo()
        chess.predict_elo()


if __name__ == "__main__":
    main()
