"""
Graph Neural Network (GNN) skeleton for spatio-temporal & attribute graphs.
This module provides a lightweight message-passing network using plain PyTorch.
For production performance, consider using PyTorch Geometric.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.msg = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: (N, D), edge_index: (2, E) with source,dest indices
        src, dst = edge_index
        msgs = self.msg(x[src])
        agg = torch.zeros_like(x).scatter_add_(0, dst.unsqueeze(-1).expand(-1, msgs.size(-1)), msgs)
        out = self.linear(x) + agg
        return F.relu(out)


class SimpleGNN(nn.Module):
    def __init__(self, node_feat_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        layers = []
        d_in = node_feat_dim
        for i in range(num_layers):
            layers.append(SimpleGNNLayer(d_in, hidden_dim))
            d_in = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = node_feats
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.readout(h)

