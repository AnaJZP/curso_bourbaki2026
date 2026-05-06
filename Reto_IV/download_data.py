"""
download_data.py -- Descarga los datos del Reto IV.

Descarga el dataset Dynamic Pricing de Kaggle usando kagglehub.

Ejecutar UNA VEZ antes de correr main.py:

    python download_data.py
"""

import shutil
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

DEST_FILE = "dynamic_pricing.csv"


def download():
    """Descarga el CSV de Dynamic Pricing desde Kaggle."""
    dest = DATA_DIR / DEST_FILE
    if dest.exists():
        print(f"  Ya existe: {DEST_FILE} ({dest.stat().st_size:,} bytes)")
        return

    try:
        import kagglehub
        path = kagglehub.dataset_download("arashnic/dynamic-pricing-dataset")
        src = Path(path) / DEST_FILE
        if src.exists():
            shutil.copy(src, dest)
            print(f"  Guardado: {dest} ({dest.stat().st_size:,} bytes)")
        else:
            print(f"  Error: No se encontro {DEST_FILE} en {path}")
    except ImportError:
        print("  kagglehub no instalado. Instalar con: pip install kagglehub")
        print("  O descargar manualmente desde:")
        print("  https://www.kaggle.com/datasets/arashnic/dynamic-pricing-dataset")


if __name__ == "__main__":
    dest = DATA_DIR / DEST_FILE
    if dest.exists():
        print(f"El archivo ya existe en data/{DEST_FILE}.")
        print("Para re-descargar, elimina el archivo de data/.")
    else:
        download()
        print("\nDescarga completada.")
