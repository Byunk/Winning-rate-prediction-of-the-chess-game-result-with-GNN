import pandas as pd

NUM_TOTAL = 1350176

def parse_key(str):
    return str.replace("[", "").replace("]", "").replace("\n", "").replace("\"", "").split(" ")[0]

def parse_item(str):
    return str.replace("[", "").replace("]", "").replace("\n", "").replace("\"", "").split(" ")[1]

with open('data/lichess_db_standard_rated_2014-12.pgn') as f:
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
                result = 0
            elif result == "0-1":
                result = -1
            else:
                raise NotImplementedError
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
    df.to_csv('data/lichess_db_standard_rated_2014-12.csv')

    print("Progress Finished")