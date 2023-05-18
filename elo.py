import pandas as pd

class ELO:
	K_FACTOR = 32 # 16 for masters and 32 for weaker players
	INIT_RATE = 1500

	def __init__(self):
		self.df = pd.read_csv("data/lichess_db_standard_rated_2014-12.csv")
		self.elo = {}

	def train(self):
		for row in self.df.itertuples():
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

	def predict(self, white, black):
		if white not in self.elo or black not in self.elo:
			raise

		white_rate = self.elo[white]
		black_rate = self.elo[black]

		Q_A = 10 ** (white_rate / 400)
		Q_B = 10 ** (black_rate / 400)
		E_A = Q_A / (Q_A + Q_B)
		return E_A, 1 - E_A
