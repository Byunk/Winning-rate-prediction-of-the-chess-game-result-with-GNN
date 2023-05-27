import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class Encoder(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_node, num_layers, heads=1
    ):
        super().__init__()

        x = torch.randn(num_node, in_channels)
        self.register_buffer("x", x)

        self.num_node = num_node
        self.in_channels = in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=4)
        )
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=4)
            )
        self.convs.append(
            GATv2Conv(hidden_channels * heads, out_channels, heads=1, edge_dim=4)
        )

    def forward(self, edge_index, edge_attr):
        x = self.x
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x
