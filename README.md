# Winning rate prediction of the chess game result with GNN

-   [Winning rate prediction of the chess game result with GNN](#winning-rate-prediction-of-the-chess-game-result-with-gnn)
    -   [Problem Statement](#problem-statement)
    -   [How to Run](#how-to-run)
        -   [Setup](#setup)
        -   [Train GNN Model](#train-gnn-model)
        -   [Predict with GNN Model](#predict-with-gnn-model)
        -   [Train ELO Model \& Predict](#train-elo-model--predict)
    -   [Methodology](#methodology)
        -   [Graph Structure](#graph-structure)
        -   [Model Architecture](#model-architecture)
            -   [Encoder](#encoder)
            -   [Decoder](#decoder)
        -   [Evaluation](#evaluation)
    -   [Result](#result)
    -   [Future Works](#future-works)
        -   [Node feature initilization](#node-feature-initilization)
        -   [Using the information of match order](#using-the-information-of-match-order)
        -   [Boosting capacity of the model](#boosting-capacity-of-the-model)
    -   [Reference](#reference)
    -   [Contributors](#contributors)

## Problem Statement

Predicting the outcome of a chess game has a significant impact on research area of prediction. Statistical analysis still plays a prominent role and achieves a reasonable level of accuracy. However, one notable drawback is that ELO cannot capture the characteristics of an individual chess player due to its inability to represent the "non-transitive" relationship. For instance, if player A holds significant superiority over specific opponenets B while has lower ELO rate, ELO simply predict B is likely to win over A. This often becomes a major flaw when when predicting the game result between players.

Therefore, we propose a GATv2-based GNN model to utilize the structural information present in the dataset, such as the superiority between players, types of openings, and imbalances in win rates between black and white, in order to enhance the predictive capabilities.

Our brief solution is as below

1. Construct the graph, players as nodes, game results as edges
2. Node features denote the multi-dimensional representation of a user's ability
3. Predict unseen game results with trained node features

## How to Run

### Setup

```sh
# Download the dataset
./setup.sh

# Download the dependencies
pip install -r requirements.txt
```

### Train GNN Model

```sh
python main.py --method gnn
```

### Predict with GNN Model

```sh
python main.py --method gnn -t
```

### Train ELO Model & Predict

```sh
python main.py --method elo
```

## Methodology

### Graph Structure

<img width="576" alt="image" src="assets/graph_structure.png">

### Model Architecture

<img width="608" alt="image" src="assets/model_architecture.png">

#### Encoder

<img width="567" alt="image" src="assets/encoder.png">

#### Decoder

<img width="572" alt="image" src="assets/decoder.png">

### Evaluation

<img width="576" alt="image" src="assets/evaluation.png">

## Result

<img width="576" alt="image" src="assets/result.png">

|                           Model                            |             Evaluation Result             |
| :--------------------------------------------------------: | :---------------------------------------: |
|     GATv2 (node feature dim: 10, heads: 3, layers: 2)      |                  0.1624                   |
|     GATv2 (node feature dim: 32, heads: 4, layers: 2)      |                  0.1625                   |
|     GATv2 (node feature dim: 32, heads: 3, layers: 3)      |                  0.1616                   |
| GATv2 (node feature dim: 32, heads: 3, layers: 2, softmax) |                  0.1612                   |
|     GATv2 (node feature dim: 32, heads: 2, layers: 4)      | **<span style="color:red">0.1595</span>** |
|                     **ELO (Baseline)**                     | **<span style="color:red">0.1461</span>** |

## Future Works

### Node feature initilization

1. For now, node features are initialized randomly
2. ELO value itself captures individual chess player's skill, which might be good initial point

### Using the information of match order

1. For now, we collapse every matches between players into 2 edge features (W-B, B-W)
2. In Elo, however, it calculates the stat with respect to the order of the matches
3. Introducing, the information about match order might help our model generalized better

### Boosting capacity of the model

1. Our model converges on high training error
2. Generalization error is relatively smaller than training erroor
3. Boosting capacity of the model will work well for both encoder and decoder

## Reference

[Lichess Dataset](https://database.lichess.org/)

## Contributors

[Dongwook Shin](https://github.com/jentleshin)

[Kyungho Byoun](https://github.com/byunk)
