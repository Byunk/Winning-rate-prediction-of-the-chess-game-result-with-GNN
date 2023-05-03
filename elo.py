import pandas as pd

K_FACTOR = 32 # 16 for masters and 32 for weaker players
INITIAL = 1500

df = pd.read_csv("data/lichess_db_standard_rated_2014-12.csv")

# whites = df['white'].unique()
# blacks = df['black'].unique()
# players = (np.unique(np.concatenate((whites, blacks))

elo = {}
for row in df.itertuples():
	white = row[1]
	black = row[2]
	result = float(row[3])

	if white in elo:
		white_rate = elo[white]
	else:
		white_rate = INITIAL

	if black in elo:
		black_rate = elo[black]
	else:
		black_rate = INITIAL
	
	Q_A = 10 ** (white_rate / 400)
	Q_B = 10 ** (black_rate / 400)
	E_A = Q_A / (Q_A + Q_B)
	E_B = Q_B / (Q_A + Q_B)

	elo[white] = white_rate + K_FACTOR * (result - E_A)
	elo[black] = black_rate + K_FACTOR * (1 - result - E_B)
