import torch
import torch.nn as nn


class ProjectionHead(nn.Module):

    def __init__(self, input_dim=384, hidden_dim=256, output_dim=128, num_layers=2):
        super().__init__()
        layers = []

        in_d = input_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.GELU())
            in_d = hidden_dim

        layers.append(nn.Linear(in_d, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        z = self.mlp(x)
        return nn.functional.normalize(z, dim=-1)
