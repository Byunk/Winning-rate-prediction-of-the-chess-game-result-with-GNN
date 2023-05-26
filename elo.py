import pandas as pd

class ELO:
	K_FACTOR = 32 # 16 for masters and 32 for weaker players
	INIT_RATE = 1500

	def __init__(self, data_dir):
		self.df = pd.read_csv(data_dir)
		self.elo = {}

	def train(self, test_edge_index):
		for i,row in enumerate(self.df.itertuples()):
			if i%1000==0:
				print(f"test {i}")
			white = row[1]
			black = row[2]
			if [white, black] in test_edge_index.tolist():
				continue
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

	def predict(self, test_edge_index):
		outputs=[]
		for i, [white, black] in enumerate(test_edge_index.tolist()):
			if i%1000==0:
				print(f"test {i}")
			if white not in self.elo or black not in self.elo:
				raise
			white_rate = self.elo[white]
			black_rate = self.elo[black]

			Q_A = 10 ** (white_rate / 400)
			Q_B = 10 ** (black_rate / 400)
			E_A = Q_A / (Q_A + Q_B)
			outputs.append([E_A, 0, 1 - E_A])
		return outputs