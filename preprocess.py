import pandas as pd
import torch

NUM_TOTAL = 1350176

def parse_key(str):
    return str.replace("[", "").replace("]", "").replace("\n", "").replace("\"", "").split(" ")[0]

def parse_item(str):
    return str.replace("[", "").replace("]", "").replace("\n", "").replace("\"", "").split(" ")[1]

def preprocess_to_dataframe(filename = 'data/lichess_db_standard_rated_2014-12.pgn'):
    with open(filename) as f:
        white_list = []
        black_list = []
        result_list = []

        iter = 0
        for line in f:
            key = parse_key(line)
            if key == "White":
                white_list.append(parse_item(line))
            elif key == "Black":
                black_list.append(parse_item(line))
            elif key == "Result":
                result = parse_item(line)
                if result == "1-0":
                    result = 1
                elif result == "1/2-1/2":
                    result = 0.5
                elif result == "0-1":
                    result = 0
                result_list.append(result)
                
                iter += 1
                if iter % 10000 == 0:
                    progress = round(((iter / NUM_TOTAL) * 100), 2)
                    print("In progress... {:.2f}%".format(progress))

        data = {
            "white": white_list,
            "black": black_list,
            "result": result_list
        }
        df = pd.DataFrame(data)
        df.to_csv('data/lichess_db_standard_rated_2014-12.csv', index=False)

        print("Progress Finished")

def preprocess_to_torch_model(filename = 'data/lichess_db_standard_rated_2014-12.csv'):
    df = pd.read_csv("../data/lichess_db_standard_rated_2014-12.csv")

    # Node list
    node_list = pd.concat([df['white'], df['black']]).unique().tolist()
    num_node = len(node_list)
    node_dict = {player_id: node for node, player_id in enumerate(node_list)}

    # Initialize edge dictionary
    edge_dict = {}

    # Iterate over dataframe rows
    for idx, row in df.iterrows():
        white = node_dict[row['white']]
        black = node_dict[row['black']]
        edge = (white, black)
        
        if edge not in edge_dict:
            edge_dict[edge] = {"win": 0, "lose": 0, "draw": 0}
        
        if row['result'] == 1:
            edge_dict[edge]["win"] += 1
        elif row['result'] == 0:
            edge_dict[edge]["lose"] += 1
        elif row['result'] == 0.5:
            edge_dict[edge]["draw"] += 1

    # Create edge_index tensor
    edge_index = torch.tensor([[white,black] for (white, black), _ in edge_dict.items()], dtype=torch.long).t().contiguous()
    # Create edge_attr tensor
    edge_attr = torch.tensor([[stats['win'], stats['draw'], stats['lose']] for _, stats in edge_dict.items()], dtype=torch.float)

    torch.save({'edge_index': edge_index, 'edge_attr': edge_attr, 'num_node':num_node}, 'data/graph_data.pt')