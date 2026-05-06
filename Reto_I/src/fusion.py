"""
fusion.py — Estrategias de fusión multi-modal.

Estrategias implementadas
-------------------------
* **concat**   : z = [t ‖ m]              → dim 1024
* **hadamard** : z = t ⊙ m                → dim 512
* **combined** : z = [t ‖ m ‖ t ⊙ m]      → dim 1536
"""

import numpy as np

from .config import FUSION_STRATEGIES


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalización L2 por fila."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def fuse(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
    strategy: str = "concat",
) -> np.ndarray:
    """Aplica una estrategia de fusión sobre embeddings normalizados."""
    t = _normalize(text_emb)
    m = _normalize(image_emb)

    if strategy == "concat":
        return np.concatenate([t, m], axis=1)
    elif strategy == "hadamard":
        return t * m
    elif strategy == "combined":
        return np.concatenate([t, m, t * m], axis=1)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")


def fuse_all(
    text_emb: np.ndarray,
    image_emb: np.ndarray,
) -> dict[str, np.ndarray]:
    """Genera todas las fusiones disponibles."""
    fusions = {}
    for strat in FUSION_STRATEGIES:
        fusions[strat] = fuse(text_emb, image_emb, strategy=strat)
    return fusions
