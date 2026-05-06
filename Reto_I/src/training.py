"""
training.py — Loop de entrenamiento y evaluación.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .config import (
    BATCH_SIZE,
    DEVICE,
    DROPOUT,
    EPOCHS,
    HIDDEN_DIM,
    LR,
    SEED,
    TEST_SIZE,
    WEIGHT_DECAY,
)
from .model import MemeDataset, MultiModalClassifier


@dataclass
class TrainResult:
    """Contenedor de resultados de un experimento."""

    strategy: str
    model: MultiModalClassifier
    history: dict = field(default_factory=dict)
    preds: np.ndarray = field(default_factory=lambda: np.array([]))
    labels: np.ndarray = field(default_factory=lambda: np.array([]))


def train_model(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> tuple[MultiModalClassifier, dict, np.ndarray, np.ndarray]:
    """Entrena el clasificador y devuelve modelo, historial, predicciones y labels."""

    train_loader = DataLoader(
        MemeDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        MemeDataset(X_val, y_val), batch_size=BATCH_SIZE
    )

    model = MultiModalClassifier(input_dim, HIDDEN_DIM, DROPOUT).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_auroc": []}

    for epoch in range(EPOCHS):
        # ── Train ────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for feats, lbls in train_loader:
            feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(feats).squeeze()
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # ── Validation ───────────────────────────────────────
        model.eval()
        val_preds, val_labels, val_loss_sum = [], [], 0.0
        with torch.no_grad():
            for feats, lbls in val_loader:
                feats, lbls = feats.to(DEVICE), lbls.to(DEVICE)
                logits = model(feats).squeeze()
                val_loss_sum += criterion(logits, lbls).item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())

        auroc = roc_auc_score(val_labels, val_preds)
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_loss"].append(val_loss_sum / len(val_loader))
        history["val_auroc"].append(auroc)
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1:3d}/{EPOCHS} │ "
                f"Train Loss: {history['train_loss'][-1]:.4f} │ "
                f"Val Loss: {history['val_loss'][-1]:.4f} │ "
                f"AUROC: {auroc:.4f}"
            )

    return model, history, np.array(val_preds), np.array(val_labels)


def run_experiments(
    fusions: dict[str, np.ndarray],
    labels: np.ndarray,
) -> dict[str, TrainResult]:
    """Entrena un modelo por cada estrategia de fusión."""
    results: dict[str, TrainResult] = {}

    for strat_name, X in fusions.items():
        print(f"\n{'=' * 50}")
        print(f"Entrenando con fusión: {strat_name.upper()}")
        print(f"{'=' * 50}")

        X_tr, X_val, y_tr, y_val = train_test_split(
            X, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels
        )
        mdl, hist, preds, true_l = train_model(
            X_tr, X_val, y_tr, y_val, input_dim=X.shape[1]
        )
        results[strat_name] = TrainResult(
            strategy=strat_name,
            model=mdl,
            history=hist,
            preds=preds,
            labels=true_l,
        )

    return results
