# Winning rate prediction of the chess game result with GNN

## Problem Statement

Predicting the outcome of a chess game has a significant impact on research area of prediction. Statistical analysis still plays a prominent role and achieves a reasonable level of accuracy. However, one notable drawback is that ELO cannot capture the characteristics of an individual chess player due to its inability to represent the "non-transitive" relationship. For instance, if player A holds significant superiority over specific opponenets B while has lower ELO rate, ELO simply predict B is likely to win over A. This often becomes a major flaw when when predicting the game result between players.

Therefore, we propose a GATv2-based GNN model to utilize the structural information present in the dataset, such as the superiority between players, types of openings, and imbalances in win rates between black and white, in order to enhance the predictive capabilities.

Our brief solution is as below

1. Construct the graph, players as nodes, game results as edges
2. Randomly initilize node features
3. Update node features with GNN
4. Randomly select edges, and calculate the loss with real game results
5. Predict unseen game results with trained node features

## Methodology

### Graph Structure

<img width="576" alt="image" src="https://github.com/Byunk/Graph_based_Chess_result_prediction/assets/60650372/83dd2f3b-0100-4701-acd1-2ac9c9b5f10f">

### Model Architecture

<img width="608" alt="image" src="https://github.com/Byunk/Graph_based_Chess_result_prediction/assets/60650372/85e15afc-299b-49e2-8cdc-97fb1e91b8ee">

#### Encoder 

<img width="567" alt="image" src="https://github.com/Byunk/Graph_based_Chess_result_prediction/assets/60650372/ada9c475-c98b-4438-b272-9bfba6fdcd2c">

#### Decoder

<img width="572" alt="image" src="https://github.com/Byunk/Graph_based_Chess_result_prediction/assets/60650372/fe6463f2-24bb-4ed0-8e02-d5359442fcba">

### Evaluation

<img width="576" alt="image" src="https://github.com/Byunk/Graph_based_Chess_result_prediction/assets/60650372/adb63c33-5718-4629-a4d2-2556122bb604">

## Result

## Future Works

<img width="576" alt="image" src="https://github.com/Byunk/Graph_based_Chess_result_prediction/assets/60650372/1579e9e3-81cb-407f-87f7-32e1f13c26c1">

## Reference

[Lichess Dataset](https://database.lichess.org/)

## Contributors

[Dongwook Shin](https://github.com/jentleshin)

[Kyungho Byoun](https://github.com/byunk)

