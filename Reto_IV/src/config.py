"""
config.py -- Configuracion global del Reto IV.
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
DATA_FILE = "dynamic_pricing.csv"

# -- Acciones (multiplicadores de precio) ----------------------
# El agente elige un multiplicador sobre el precio base estimado
PRICE_MULTIPLIERS = [0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.35, 1.50]
N_ACTIONS = len(PRICE_MULTIPLIERS)

# -- Hiperparametros DQN --------------------------------------
DQN_HIDDEN     = 128
DQN_LR         = 1e-3
DQN_GAMMA      = 0.95      # factor de descuento
DQN_EPS_START  = 1.0       # epsilon inicial
DQN_EPS_END    = 0.05      # epsilon final
DQN_EPS_DECAY  = 0.995     # decaimiento por episodio
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 10000
DQN_TARGET_UPDATE = 10     # actualizar target network cada N episodios
DQN_EPISODES   = 500
DQN_STEPS_PER_EP = 50      # pasos por episodio

# -- Paleta de colores (escala de azules) ---------------------
COLORS = {
    "fixed":        "#1B4F72",
    "proportional": "#2471A3",
    "dqn":          "#2E86C1",
}
COLOR_PRIMARY   = "#2E86C1"
COLOR_SECONDARY = "#1B4F72"
COLOR_ACCENT    = "#85C1E9"
COLOR_LIGHT     = "#AED6F1"
CMAP_HEATMAP    = "Blues"

# -- Suprimir warnings ----------------------------------------
warnings.filterwarnings("ignore")
