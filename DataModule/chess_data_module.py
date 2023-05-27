import pandas as pd
import torch
import os
import os.path

DATA_DIR = "data/"
SOURCE_NAME = "lichess_db_standard_rated_2014-12.pgn"


class ChessData:
    def __init__(self, dir=SOURCE_NAME):
        self.dir = DATA_DIR + dir
        self.csv_dir = self.change_extension("csv")
        self.pt_dir = self.change_extension("pt")

    def prepare(self):
        # Preprocessed files already exist
        if os.path.exists(os.getcwd() + self.csv_dir) and os.path.exists(
            os.getcwd() + self.pt_dir
        ):
            return

        print("Start converting raw to df...")
        self.raw_to_df()
        print("Start converting df to dict...")
        self.df_to_dict()

        if not os.path.exists(os.getcwd() + self.csv_dir):
            print("Start saving csv...")
            self.dict_to_csv()

        if not os.path.exists(os.getcwd() + self.pt_dir):
            print("Start saving tensor...")
            self.dict_to_tensor()

    def raw_to_df(self):
        with open(self.dir) as f:
            white_list = []
            black_list = []
            result_list = []

            n_row = 0
            for line in f:
                key = self.parse_key(line)
                if key == "White":
                    item = self.parse_item(line)
                    white_list.append(item)
                elif key == "Black":
                    item = self.parse_item(line)
                    black_list.append(item)
                elif key == "Result":
                    item = self.parse_item(line)
                    if item == "1-0":
                        result = 1
                    elif item == "1/2-1/2":
                        result = 0.5
                    elif item == "0-1":
                        result = 0
                    result_list.append(result)

                    n_row += 1
                    if n_row % 100000 == 0:
                        print("{0} rows save into df".format(n_row))

            data = {"white": white_list, "black": black_list, "result": result_list}
            self.df = pd.DataFrame(data)

    def df_to_dict(self):
        node_list = pd.concat([self.df["white"], self.df["black"]]).unique().tolist()
        self.node_dict = {player_id: node for node, player_id in enumerate(node_list)}
        self.edge_dict = {}
        self.num_node = len(node_list)

        for _, row in self.df.iterrows():
            white = self.node_dict[row["white"]]
            black = self.node_dict[row["black"]]
            edge = (white, black)

            if edge not in self.edge_dict:
                self.edge_dict[edge] = {"win": 0, "lose": 0, "draw": 0}

            if row["result"] == 1:
                self.edge_dict[edge]["win"] += 1
            elif row["result"] == 0:
                self.edge_dict[edge]["lose"] += 1
            elif row["result"] == 0.5:
                self.edge_dict[edge]["draw"] += 1

    def dict_to_csv(self):
        self.df["white"] = self.df["white"].map(self.node_dict)
        self.df["black"] = self.df["black"].map(self.node_dict)
        self.df.to_csv(self.csv_dir, index=False)

    def dict_to_tensor(self):
        edge_index = (
            torch.tensor(
                [[white, black] for (white, black), _ in self.edge_dict.items()],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )
        edge_attr = torch.tensor(
            [
                [stats["win"], stats["draw"], stats["lose"]]
                for _, stats in self.edge_dict.items()
            ],
            dtype=torch.float,
        )

        torch.save(
            {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "num_node": self.num_node,
            },
            self.pt_dir,
        )

    def parse_key(self, str):
        str_list = (
            str.replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace('"', "")
            .split(" ")
        )
        return str_list[0]

    def parse_item(self, str):
        str_list = (
            str.replace("[", "")
            .replace("]", "")
            .replace("\n", "")
            .replace('"', "")
            .split(" ")
        )
        return str_list[1]

    def change_extension(self, ext):
        dir_list = self.dir.split(".")
        dir_list[1] = ext
        return ".".join(dir_list)
