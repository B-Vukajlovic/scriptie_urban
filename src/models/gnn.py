"""Small PyTorch GNN regressors for block-level accessibility prediction."""

from __future__ import annotations

import torch
from torch import nn


class GCNLayer(nn.Module):
    """Graph convolution layer using pre-normalized adjacency."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.linear(torch.sparse.mm(adjacency, x))


class GraphSAGELayer(nn.Module):
    """Mean GraphSAGE layer with learned self and neighbor representations."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, neighbor_adjacency: torch.Tensor) -> torch.Tensor:
        neighbor_mean = torch.sparse.mm(neighbor_adjacency, x)
        return self.linear(torch.cat([x, neighbor_mean], dim=1))


class GCNRegressor(nn.Module):
    """Two-layer GCN for node-level regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.conv1(x, adjacency)))
        x = self.dropout(self.activation(self.conv2(x, adjacency)))
        return self.output(x).squeeze(-1)


class GraphSAGERegressor(nn.Module):
    """Two-layer mean GraphSAGE model for node-level regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.sage1 = GraphSAGELayer(in_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, neighbor_adjacency: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.sage1(x, neighbor_adjacency)))
        x = self.dropout(self.activation(self.sage2(x, neighbor_adjacency)))
        return self.output(x).squeeze(-1)
