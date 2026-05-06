"""
model.py — Arquitectura del clasificador multi-modal y Dataset de PyTorch.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MultiModalClassifier(nn.Module):
    """
    Red feedforward de 3 capas con BatchNorm, GELU y Dropout.

    Arquitectura
    ------------
    Linear(input_dim → hidden_dim) → BN → GELU → Dropout
    Linear(hidden_dim → hidden_dim//2) → BN → GELU → Dropout
    Linear(hidden_dim//2 → 1)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class MemeDataset(Dataset):
    """Dataset sencillo que envuelve arrays de features y labels."""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
