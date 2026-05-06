"""
download_data.py — Descarga el dataset Hateful Memes desde HuggingFace.
Ejecutar UNA VEZ antes de correr main.py:

    python download_data.py
"""

import io
import os
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

# ── Configuracion ────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET = "neuralcatcher/hateful_memes"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"


def download_dataset():
    """Descarga metadata via datasets lib e imagenes via HTTP."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Instalando libreria datasets...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset

    print(f"Descargando metadata del dataset: {HF_DATASET}")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"  Total de muestras en HuggingFace: {len(ds)}")

    records = []
    downloaded = 0
    skipped = 0
    failed = 0

    for item in tqdm(ds, desc="Descargando imagenes"):
        original_id = item["id"]
        text = item["text"]
        label = item["label"]
        img_ref = item["img"]  # e.g. 'img/42953.png'

        # Nombre local del archivo
        oid_str = str(original_id).zfill(5)
        img_path = IMG_DIR / f"{oid_str}.png"

        # Si ya existe, solo registrar
        if img_path.exists():
            skipped += 1
            records.append({
                "id": len(records),
                "original_id": original_id,
                "text": text,
                "label": label,
                "img_path": str(img_path),
            })
            continue

        # Descargar imagen desde el repo de HuggingFace
        img_url = f"{HF_REPO_URL}/{img_ref}"
        try:
            resp = requests.get(img_url, timeout=20)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            img.save(img_path)
            downloaded += 1
            records.append({
                "id": len(records),
                "original_id": original_id,
                "text": text,
                "label": label,
                "img_path": str(img_path),
            })
        except Exception as e:
            failed += 1

    # Guardar CSV
    df = pd.DataFrame(records)
    csv_path = DATA_DIR / "data.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'=' * 50}")
    print(f"Descarga completada:")
    print(f"  Nuevas descargadas: {downloaded}")
    print(f"  Ya existian:        {skipped}")
    print(f"  Fallidas:           {failed}")
    print(f"  Total en CSV:       {len(records)}")
    print(f"  CSV guardado:       {csv_path}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    csv_path = DATA_DIR / "data.csv"
    n_images = len(list(IMG_DIR.glob("*.png")))
    if csv_path.exists() and n_images > 1000:
        print(f"El dataset ya existe ({n_images} imagenes).")
        print("Para re-descargar, elimina data/data.csv")
    else:
        download_dataset()
