"""
config.py -- Configuracion global del Reto III.
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
LISTING_FILE = "listings.csv"

# Features numericas a usar
NUMERIC_FEATURES = [
    "accommodates",
    "bedrooms",
    "beds",
    "minimum_nights",
    "maximum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_cleanliness",
    "review_scores_location",
    "review_scores_value",
    "availability_365",
    "calculated_host_listings_count",
    "host_listings_count",
    "latitude",
    "longitude",
]

# Features de texto para embeddings
TEXT_FEATURES = ["name", "description"]

# -- Grafo ----------------------------------------------------
K_NEIGHBORS       = 10     # KNN para grafo geografico
COSINE_THRESHOLD   = 0.80   # umbral para grafo textual
TEXT_EMBED_MODEL   = "all-MiniLM-L6-v2"

# -- Hiperparametros GCN -------------------------------------
GCN_HIDDEN   = 64
GCN_DROPOUT  = 0.3
GCN_EPOCHS   = 300
GCN_LR       = 0.01
GCN_WD       = 5e-4

# -- Split ----------------------------------------------------
TEST_RATIO = 0.15
VAL_RATIO  = 0.15

# -- Paleta de colores (escala de azules) ---------------------
COLORS = {
    "linear":  "#1B4F72",
    "gcn_geo": "#2E86C1",
    "gcn_txt": "#85C1E9",
}
COLOR_PRIMARY   = "#2E86C1"
COLOR_SECONDARY = "#1B4F72"
COLOR_ACCENT    = "#85C1E9"
CMAP_HEATMAP    = "Blues"

# -- Suprimir warnings ----------------------------------------
warnings.filterwarnings("ignore")
