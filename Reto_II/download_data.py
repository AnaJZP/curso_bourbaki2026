"""
download_data.py -- Descarga los datos del Reto II desde el repo de Bourbaki.
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

BASE_URL = ("https://raw.githubusercontent.com/pedro9olivares/Bourbaki/"
            "main/BBVA/ML-%26-AI/Datos-del-reto-II")

FILES = ["X_train.csv", "y_train.csv", "assets.csv"]


def download():
    """Descarga los archivos CSV del repo de Bourbaki."""
    for fname in FILES:
        dest = DATA_DIR / fname
        if dest.exists():
            print(f"  Ya existe: {fname}")
            continue

        url = f"{BASE_URL}/{fname}"
        print(f"  Descargando {fname} ...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  Guardado: {dest} ({len(resp.content):,} bytes)")

    print("\nDescarga completada.")


if __name__ == "__main__":
    all_exist = all((DATA_DIR / f).exists() for f in FILES)
    if all_exist:
        print("Todos los archivos ya existen en data/.")
        print("Para re-descargar, elimina los archivos de data/.")
    else:
        download()
