"""
training.py -- Entrenamiento y evaluacion de los tres modelos del Reto II.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    DEVICE,
    GROUPS,
    LSTM_BATCH,
    LSTM_EPOCHS,
    LSTM_LR,
    SEED,
    VAL_WEEKS,
)
from .model import BaselineModel, ProductionLSTM, create_gbr


@dataclass
class ModelResult:
    """Contenedor de resultados por modelo."""
    name: str
    mse_per_group: dict = field(default_factory=dict)
    preds_per_group: dict = field(default_factory=dict)
    true_per_group: dict = field(default_factory=dict)
    sample_ids_val: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_importance: dict = field(default_factory=dict)
    history: dict = field(default_factory=dict)


def temporal_split(sample_ids: np.ndarray, val_weeks: int = VAL_WEEKS):
    """Split temporal: primeras N-val_weeks para train, ultimas val_weeks para val."""
    sorted_ids = np.sort(sample_ids)
    split_point = len(sorted_ids) - val_weeks
    train_ids = sorted_ids[:split_point]
    val_ids = sorted_ids[split_point:]
    return train_ids, val_ids


# ── Baseline ─────────────────────────────────────────────────

def train_baseline(
    pivot: pd.DataFrame,
    y: pd.DataFrame,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
) -> ModelResult:
    """Entrena y evalua el modelo baseline (anomaly detection on/off)."""
    result = ModelResult(name="baseline")
    result.sample_ids_val = val_ids

    model = BaselineModel(threshold=2.0)

    # Fit con datos de train
    pivot_train = pivot[pivot["SAMPLE_ID"].isin(train_ids)]
    model.fit(pivot_train)

    # Predecir en validacion
    pivot_val = pivot[pivot["SAMPLE_ID"].isin(val_ids)]

    for g in GROUPS:
        preds = []
        trues = []
        target_col = f"PRODUCTION_GROUP_{g}"

        for sid in val_ids:
            week_data = pivot_val[pivot_val["SAMPLE_ID"] == sid]
            pred = model.predict_group_production(week_data, g)
            true_val = y[y["SAMPLE_ID"] == sid][target_col].values[0]
            preds.append(pred)
            trues.append(true_val)

        preds = np.array(preds)
        trues = np.array(trues)
        mse = mean_squared_error(trues, preds)
        result.mse_per_group[g] = mse
        result.preds_per_group[g] = preds
        result.true_per_group[g] = trues
        print(f"  Baseline -- Grupo {g}: MSE = {mse:.2e}")

    return result


# ── GradientBoosting ─────────────────────────────────────────

def train_gbr(
    group_features: dict,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
) -> ModelResult:
    """Entrena y evalua GradientBoosting por grupo."""
    result = ModelResult(name="gbr")
    result.sample_ids_val = val_ids

    for g in GROUPS:
        X_df, y_vals, sample_ids = group_features[g]

        train_mask = np.isin(sample_ids, train_ids)
        val_mask = np.isin(sample_ids, val_ids)

        X_train = X_df[train_mask].values
        y_train = y_vals[train_mask]
        X_val = X_df[val_mask].values
        y_val = y_vals[val_mask]

        gbr = create_gbr()
        gbr.fit(X_train, y_train)

        preds = gbr.predict(X_val)
        mse = mean_squared_error(y_val, preds)

        result.mse_per_group[g] = mse
        result.preds_per_group[g] = preds
        result.true_per_group[g] = y_val

        # Feature importance
        result.feature_importance[g] = dict(
            zip(X_df.columns, gbr.feature_importances_)
        )

        print(f"  GBR -- Grupo {g}: MSE = {mse:.2e}")

    return result


# ── LSTM ─────────────────────────────────────────────────────

def train_lstm(
    group_features: dict,
    train_ids: np.ndarray,
    val_ids: np.ndarray,
) -> ModelResult:
    """
    Entrena un LSTM por grupo.

    NOTA: Con ~80 muestras de entrenamiento, el LSTM tiene alto riesgo de
    sobreajuste. Se incluye como comparacion pedagogica. La secuencia se
    construye con ventanas deslizantes de las ultimas 4 semanas.
    """
    result = ModelResult(name="lstm")
    result.sample_ids_val = val_ids
    result.history = {g: {"train_loss": [], "val_loss": []} for g in GROUPS}

    SEQ_LEN = 4  # ventana de 4 semanas

    for g in GROUPS:
        X_df, y_vals, sample_ids = group_features[g]

        n_features = X_df.shape[1]
        X_all = X_df.values.astype(np.float32)
        y_all = y_vals.astype(np.float32)

        # Normalizar features
        mean_x = X_all.mean(axis=0)
        std_x = X_all.std(axis=0)
        std_x[std_x == 0] = 1.0
        X_norm = (X_all - mean_x) / std_x

        # Normalizar target
        mean_y = y_all.mean()
        std_y = y_all.std()
        if std_y == 0:
            std_y = 1.0
        y_norm = (y_all - mean_y) / std_y

        # Crear secuencias
        X_seq, y_seq, seq_ids = [], [], []
        for i in range(SEQ_LEN, len(X_norm)):
            X_seq.append(X_norm[i - SEQ_LEN:i])
            y_seq.append(y_norm[i])
            seq_ids.append(sample_ids[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        seq_ids = np.array(seq_ids)

        train_mask = np.isin(seq_ids, train_ids)
        val_mask = np.isin(seq_ids, val_ids)

        X_train_t = torch.FloatTensor(X_seq[train_mask]).to(DEVICE)
        y_train_t = torch.FloatTensor(y_seq[train_mask]).to(DEVICE)
        X_val_t = torch.FloatTensor(X_seq[val_mask]).to(DEVICE)
        y_val_t = torch.FloatTensor(y_seq[val_mask]).to(DEVICE)

        train_ds = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH, shuffle=False)

        model = ProductionLSTM(input_dim=n_features).to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LSTM_LR)

        for epoch in range(LSTM_EPOCHS):
            model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            result.history[g]["train_loss"].append(epoch_loss / max(len(train_loader), 1))
            result.history[g]["val_loss"].append(val_loss)

            if (epoch + 1) % 25 == 0:
                print(f"  LSTM G{g} Epoch {epoch + 1:3d}/{LSTM_EPOCHS} | "
                      f"Train: {result.history[g]['train_loss'][-1]:.4f} | "
                      f"Val: {val_loss:.4f}")

        # Desnormalizar predicciones
        model.eval()
        with torch.no_grad():
            preds_norm = model(X_val_t).cpu().numpy()
        preds = preds_norm * std_y + mean_y
        trues = y_seq[val_mask] * std_y + mean_y

        mse = mean_squared_error(trues, preds)
        result.mse_per_group[g] = mse
        result.preds_per_group[g] = preds
        result.true_per_group[g] = trues
        print(f"  LSTM -- Grupo {g}: MSE = {mse:.2e}")

    return result


# ── Orquestador ──────────────────────────────────────────────

def run_all_models(
    pivot: pd.DataFrame,
    y: pd.DataFrame,
    group_features: dict,
) -> dict[str, ModelResult]:
    """Entrena los tres modelos y retorna resultados."""
    sample_ids = np.sort(pivot["SAMPLE_ID"].unique())
    train_ids, val_ids = temporal_split(sample_ids)

    print(f"  Split temporal: train={len(train_ids)} semanas, "
          f"val={len(val_ids)} semanas")
    print(f"  Train: semanas {train_ids[0]}-{train_ids[-1]}")
    print(f"  Val:   semanas {val_ids[0]}-{val_ids[-1]}")

    results = {}

    print("\n  --- Baseline (Anomaly Detection) ---")
    results["baseline"] = train_baseline(pivot, y, train_ids, val_ids)

    print("\n  --- GradientBoosting Regressor ---")
    results["gbr"] = train_gbr(group_features, train_ids, val_ids)

    print("\n  --- LSTM ---")
    results["lstm"] = train_lstm(group_features, train_ids, val_ids)

    return results
