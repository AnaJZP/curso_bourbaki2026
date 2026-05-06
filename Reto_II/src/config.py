"""
config.py -- Configuracion global del Reto II.
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
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = PROJECT_DIR / "data"
OUTPUT_DIR  = PROJECT_DIR / "resultados"
OUTPUT_DIR.mkdir(exist_ok=True)

# -- Datos ----------------------------------------------------
N_ASSETS       = 83
N_MEASURE_TYPES = 4
N_WEEKDAYS     = 7
GROUPS         = [2, 3]

# -- Hiperparametros GradientBoosting -------------------------
GB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "min_samples_leaf": 5,
    "random_state": SEED,
}

# -- Hiperparametros LSTM ------------------------------------
LSTM_HIDDEN   = 64
LSTM_LAYERS   = 2
LSTM_DROPOUT  = 0.2
LSTM_EPOCHS   = 100
LSTM_LR       = 1e-3
LSTM_BATCH    = 16

# -- Split temporal -------------------------------------------
VAL_WEEKS = 24  # ultimas 24 semanas para validacion

# -- Paleta de colores (escala de azules) ---------------------
COLORS = {
    "baseline": "#1B4F72",
    "gbr":      "#2E86C1",
    "lstm":     "#85C1E9",
}
COLOR_GROUP_2 = "#2E86C1"
COLOR_GROUP_3 = "#1B4F72"
CMAP_HEATMAP  = "Blues"

# -- Suprimir warnings ----------------------------------------
warnings.filterwarnings("ignore")
