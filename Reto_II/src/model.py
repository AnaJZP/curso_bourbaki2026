"""
model.py -- Modelos para el Reto II.

Tres enfoques:
1. Baseline (anomaly detection on/off)
2. GradientBoosting Regressor
3. LSTM (comparacion, con limitaciones documentadas)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor

from .config import (
    DEVICE,
    GB_PARAMS,
    LSTM_DROPOUT,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    SEED,
)


# ── Baseline: Anomaly Detection ──────────────────────────────

class BaselineModel:
    """
    Baseline sugerido por el challenge:
    - Si las mediciones de un activo estan cerca de 0, el activo esta 'on'
      y produce a capacidad nominal.
    - Si hay anomalias (mediciones alejadas del centroide), el activo esta 'off'
      y produce 0.
    """

    def __init__(self, threshold: float = 2.0):
        self.threshold = threshold
        self.centroids = {}  # {asset_id: centroid_vector}
        self.stds = {}

    def fit(self, pivot_train):
        """Calcula centroide y std por activo usando datos de entrenamiento."""
        mt_cols = [c for c in pivot_train.columns if c.startswith("mt") and "_wd" in c]
        for asset_id in pivot_train["ASSET_ID"].unique():
            asset_data = pivot_train[pivot_train["ASSET_ID"] == asset_id][mt_cols].values
            self.centroids[asset_id] = np.nanmean(asset_data, axis=0)
            std = np.nanstd(asset_data, axis=0)
            std[std == 0] = 1.0  # evitar division por cero
            self.stds[asset_id] = std

    def predict_capacity_factor(self, pivot_row) -> float:
        """Predice factor de capacidad (0 o 1) para un activo-semana."""
        asset_id = pivot_row["ASSET_ID"]
        mt_cols = [c for c in pivot_row.index if c.startswith("mt") and "_wd" in c]
        values = pivot_row[mt_cols].values.astype(float)

        if asset_id not in self.centroids:
            return 1.0  # asumir on si no hay datos

        centroid = self.centroids[asset_id]
        std = self.stds[asset_id]

        # Distancia normalizada al centroide
        diff = np.abs(values - centroid)
        z_scores = diff / std
        mean_z = np.nanmean(z_scores)

        return 0.0 if mean_z > self.threshold else 1.0

    def predict_group_production(self, pivot_week, group_id: int) -> float:
        """Predice produccion agregada de un grupo para una semana."""
        group_data = pivot_week[pivot_week["GROUP_ID"] == group_id]
        total = 0.0
        for _, row in group_data.iterrows():
            cf = self.predict_capacity_factor(row)
            total += cf * row["ASSET_NOMINAL_CAPACITY"]
        return total


# ── GradientBoosting Regressor ────────────────────────────────

def create_gbr() -> GradientBoostingRegressor:
    """Crea un GradientBoostingRegressor con hiperparametros del config."""
    return GradientBoostingRegressor(**GB_PARAMS)


# ── LSTM ──────────────────────────────────────────────────────

class ProductionLSTM(nn.Module):
    """
    LSTM para prediccion de produccion semanal.

    LIMITACION IMPORTANTE: Con solo ~80 semanas de entrenamiento, este modelo
    tiene alto riesgo de sobreajuste. Se incluye como comparacion pedagogica
    para demostrar que modelos mas complejos no siempre superan a los simples
    con datos escasos.
    """

    def __init__(self, input_dim: int, hidden_dim: int = LSTM_HIDDEN,
                 n_layers: int = LSTM_LAYERS, dropout: float = LSTM_DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Tomar solo el ultimo timestep
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)
