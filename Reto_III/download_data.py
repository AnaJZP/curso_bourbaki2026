"""
download_data.py -- Descarga los datos del Reto III.

Descarga el CSV de listings de Airbnb en Santorini desde el repositorio
de referencia del paper.

Ejecutar UNA VEZ antes de correr main.py:

    python download_data.py
"""

import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import requests

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Fuente: repositorio del paper sobre prediccion de precios con GNN
SOURCE_URL = (
    "https://raw.githubusercontent.com/nkanak/predicting-prices-of-airbnb-listings"
    "/main/santorini_listings.csv"
)

DEST_FILE = "listings.csv"


def download():
    """Descarga el CSV de listings de Santorini."""
    dest = DATA_DIR / DEST_FILE
    if dest.exists():
        print(f"  Ya existe: {DEST_FILE} ({dest.stat().st_size:,} bytes)")
        return

    print(f"  Descargando {DEST_FILE} ...")
    resp = requests.get(SOURCE_URL, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"  Guardado: {dest} ({len(resp.content):,} bytes)")


if __name__ == "__main__":
    dest = DATA_DIR / DEST_FILE
    if dest.exists():
        print(f"El archivo ya existe en data/{DEST_FILE}.")
        print("Para re-descargar, elimina el archivo de data/.")
    else:
        download()
        print("\nDescarga completada.")
