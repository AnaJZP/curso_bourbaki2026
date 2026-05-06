"""
model.py -- Modelos para el Reto III.

Tres enfoques:
1. LinearRegression (sklearn): Baseline
2. GCN (Graph Convolutional Network): 2 capas con PyTorch Geometric
3. Comparacion con ambos tipos de grafo (geografico y textual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from .config import GCN_DROPOUT, GCN_HIDDEN


class PriceGCN(nn.Module):
    """
    Graph Convolutional Network para regresion de precios.

    Arquitectura:
        GCNConv(in, hidden) -> BatchNorm -> ReLU -> Dropout
        GCNConv(hidden, hidden) -> BatchNorm -> ReLU -> Dropout
        Linear(hidden, 1)

    La GCN propaga informacion entre vecinos del grafo, permitiendo
    que las predicciones de un listing se informen por las caracteristicas
    de sus vecinos (geograficos o textuales).
    """

    def __init__(self, input_dim: int, hidden_dim: int = GCN_HIDDEN,
                 dropout: float = GCN_DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.fc    = nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)
        return x.squeeze(-1)
