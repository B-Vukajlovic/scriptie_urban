"""PyTorch Geometric GNN regressors for official GNNExplainer workflows."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv


class PyGGCNRegressor(nn.Module):
    """Two-layer PyG GCN for node-level accessibility regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.conv1(x, edge_index)))
        x = self.dropout(self.activation(self.conv2(x, edge_index)))
        return self.output(x).squeeze(-1)


class PyGGraphSAGERegressor(nn.Module):
    """Two-layer PyG GraphSAGE for node-level accessibility regression."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.sage1 = SAGEConv(in_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.sage1(x, edge_index)))
        x = self.dropout(self.activation(self.sage2(x, edge_index)))
        return self.output(x).squeeze(-1)
