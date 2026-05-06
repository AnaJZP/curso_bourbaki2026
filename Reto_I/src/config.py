"""
config.py -- Configuracion global del proyecto.
Centraliza rutas, semillas, hiperparametros y constantes.
"""

import warnings
from pathlib import Path

import numpy as np
import torch

# -- Reproducibilidad -----------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -- Dispositivo ----------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Rutas ----------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent   # Reto_I/
DATA_DIR    = PROJECT_DIR / "data"
IMG_DIR     = DATA_DIR / "images"
OUTPUT_DIR  = PROJECT_DIR / "resultados"
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Modelo CLIP ----------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDING_DIM   = 512

# -- Hiperparametros de entrenamiento -------------------------
EPOCHS      = 30
LR          = 1e-3
BATCH_SIZE  = 32
HIDDEN_DIM  = 256
DROPOUT     = 0.3
TEST_SIZE   = 0.2
WEIGHT_DECAY = 1e-4

# -- Estrategias de fusion disponibles ------------------------
FUSION_STRATEGIES = ["concat", "hadamard", "combined"]

# -- Paleta de colores (escala de azules) ---------------------
COLORS = {
    "concat":   "#1B4F72",   # azul oscuro
    "hadamard": "#2E86C1",   # azul medio
    "combined": "#85C1E9",   # azul claro
}

COLOR_CLASS_0 = "#2E86C1"    # No ofensivo — azul
COLOR_CLASS_1 = "#1B4F72"    # Ofensivo    — azul oscuro
CMAP_HEATMAP  = "Blues"      # para matrices de confusion

# -- Suprimir warnings ----------------------------------------
warnings.filterwarnings("ignore")
