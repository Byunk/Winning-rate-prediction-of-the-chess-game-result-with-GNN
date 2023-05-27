import pandas as pd
import torch


class ELO:
    K_FACTOR = 32  # 16 for masters and 32 for weaker players
    INIT_RATE = 1500

    def __init__(self, data_dir, test_edge_index):
        df = pd.read_csv(data_dir)

        self.train_df = self.filter_df(df, test_edge_index)
        self.test_edge_index = test_edge_index
        self.elo = {}

    def train(self):
        for row in self.train_df.itertuples():
            white = row[1]
            black = row[2]
            result = float(row[3])

            if white in self.elo:
                white_rate = self.elo[white]
            else:
                white_rate = self.INIT_RATE

            if black in self.elo:
                black_rate = self.elo[black]
            else:
                black_rate = self.INIT_RATE

            Q_A = 10 ** (white_rate / 400)
            Q_B = 10 ** (black_rate / 400)
            E_A = Q_A / (Q_A + Q_B)
            E_B = Q_B / (Q_A + Q_B)

            self.elo[white] = white_rate + self.K_FACTOR * (result - E_A)
            self.elo[black] = black_rate + self.K_FACTOR * (1 - result - E_B)

    def predict(self):
        outputs = []
        for i, [white, black] in self.test_edge_index.iterrows():
            if white not in self.elo:
                white_rate = self.INIT_RATE
            else:
                white_rate = self.elo[white]

            if black not in self.elo:
                black_rate = self.INIT_RATE
            else:
                black_rate = self.elo[black]

            Q_A = 10 ** (white_rate / 400)
            Q_B = 10 ** (black_rate / 400)
            E_A = Q_A / (Q_A + Q_B)
            outputs.append([E_A, 0, 1 - E_A])
        return torch.tensor(outputs)

    def filter_df(self, df_A, df_B):
        df_merged = df_A.merge(df_B, on=["white", "black"], how="outer", indicator=True)
        df_A = df_merged[df_merged["_merge"] == "left_only"]
        df_A = df_A.drop(columns="_merge")

        return df_A
