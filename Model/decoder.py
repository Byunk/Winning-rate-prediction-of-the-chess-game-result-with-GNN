import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=3):
        super().__init__()
        hidden1, hidden2 = hidden_channels
        self.Fc1 = nn.Linear(in_channels, hidden1, bias=False)
        self.Fc2 = nn.Linear(hidden1, hidden2, bias=False)
        self.Fc3 = nn.Linear(hidden2, out_channels, bias=False)

    def forward(self, x):
        h = F.relu(self.Fc1(x))
        h = F.relu(self.Fc2(h))
        output = self.Fc3(h)
        return output
