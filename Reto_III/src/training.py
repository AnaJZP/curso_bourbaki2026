"""
training.py -- Entrenamiento y evaluacion de los modelos del Reto III.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import (
    DEVICE,
    GCN_EPOCHS,
    GCN_LR,
    GCN_WD,
    SEED,
    TEST_RATIO,
    VAL_RATIO,
)
from .model import PriceGCN


@dataclass
class ModelResult:
    """Contenedor de resultados por modelo."""
    name: str
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0
    y_true: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    history: dict = field(default_factory=dict)


def create_masks(n: int, val_ratio: float = VAL_RATIO,
                 test_ratio: float = TEST_RATIO):
    """Crea mascaras train/val/test con split aleatorio."""
    np.random.seed(SEED)
    perm = np.random.permutation(n)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


# -- Linear Regression (Baseline) --------------------------------

def train_linear(X: np.ndarray, y: np.ndarray,
                 log_price: bool = True) -> ModelResult:
    """Entrena y evalua regresion lineal como baseline."""
    result = ModelResult(name="linear")

    train_mask, val_mask, test_mask = create_masks(len(y))

    train_idx = train_mask.numpy()
    test_idx = test_mask.numpy()

    model = LinearRegression()
    model.fit(X[train_idx], y[train_idx])

    preds = model.predict(X[test_idx])

    # Convertir de log-space a precio original para metricas
    if log_price:
        # Clipear predicciones log para evitar overflow en expm1
        preds_clipped = np.clip(preds, 0, 12)
        y_true_orig = np.expm1(y[test_idx])
        y_pred_orig = np.expm1(preds_clipped)
    else:
        y_true_orig = y[test_idx]
        y_pred_orig = preds
    y_pred_orig = np.maximum(y_pred_orig, 0)  # no permitir precios negativos

    result.y_true = y_true_orig
    result.y_pred = y_pred_orig
    result.mse = mean_squared_error(result.y_true, result.y_pred)
    result.rmse = np.sqrt(result.mse)
    result.mae = mean_absolute_error(result.y_true, result.y_pred)
    result.r2 = r2_score(result.y_true, result.y_pred)

    print(f"  Linear -- MSE={result.mse:.2f}, "
          f"MAE={result.mae:.2f}, R2={result.r2:.4f}")

    return result


# -- GCN ---------------------------------------------------------

def train_gcn(data, name: str = "gcn_geo",
              epochs: int = GCN_EPOCHS,
              log_price: bool = True) -> ModelResult:
    """Entrena GCN y evalua en test set."""
    result = ModelResult(name=name)
    result.history = {"train_loss": [], "val_loss": []}

    n = data.num_nodes
    train_mask, val_mask, test_mask = create_masks(n)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data = data.to(DEVICE)

    input_dim = data.x.shape[1]
    model = PriceGCN(input_dim=input_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=GCN_LR, weight_decay=GCN_WD)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 30
    patience_counter = 0

    for epoch in range(epochs):
        # -- Train --
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # -- Val --
        model.eval()
        with torch.no_grad():
            out_eval = model(data)
            val_loss = criterion(
                out_eval[data.val_mask], data.y[data.val_mask]
            ).item()

        result.history["train_loss"].append(loss.item())
        result.history["val_loss"].append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  {name}: Early stopping en epoch {epoch + 1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"  {name} Epoch {epoch + 1:3d}/{epochs} | "
                  f"Train: {loss.item():.2f} | Val: {val_loss:.2f}")

    # Cargar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state)

    # -- Test --
    model.eval()
    with torch.no_grad():
        preds = model(data)
        y_true_log = data.y[data.test_mask].cpu().numpy()
        y_pred_log = preds[data.test_mask].cpu().numpy()

    # Convertir de log-space a precio original
    if log_price:
        # Clipear predicciones log para evitar overflow en expm1
        y_pred_log = np.clip(y_pred_log, 0, 12)
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)
        y_pred = np.maximum(y_pred, 0)
    else:
        y_true = y_true_log
        y_pred = y_pred_log

    result.y_true = y_true
    result.y_pred = y_pred
    result.mse = mean_squared_error(y_true, y_pred)
    result.rmse = np.sqrt(result.mse)
    result.mae = mean_absolute_error(y_true, y_pred)
    result.r2 = r2_score(y_true, y_pred)

    print(f"  {name} -- MSE={result.mse:.2f}, "
          f"MAE={result.mae:.2f}, R2={result.r2:.4f}")

    return result


# -- Orquestador --------------------------------------------------

def run_all_models(X, y, geo_data, txt_data) -> dict[str, ModelResult]:
    """Entrena los tres modelos y retorna resultados."""
    n = len(y)
    train_mask, val_mask, test_mask = create_masks(n)
    n_train = train_mask.sum().item()
    n_val = val_mask.sum().item()
    n_test = test_mask.sum().item()

    print(f"  Split: train={n_train}, val={n_val}, test={n_test}")

    results = {}

    print("\n  --- Linear Regression (Baseline) ---")
    results["linear"] = train_linear(X, y)

    print("\n  --- GCN (Grafo Geografico) ---")
    results["gcn_geo"] = train_gcn(geo_data, name="gcn_geo")

    print("\n  --- GCN (Grafo Textual) ---")
    results["gcn_txt"] = train_gcn(txt_data, name="gcn_txt")

    return results
