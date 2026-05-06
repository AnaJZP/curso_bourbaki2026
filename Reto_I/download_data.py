"""
download_data.py -- Descarga el dataset Hateful Memes desde HuggingFace.
Ejecutar UNA VEZ antes de correr main.py:

    python download_data.py            # descarga 1500 imagenes (por defecto)
    python download_data.py --max 500  # descarga solo 500
    python download_data.py --max 0    # descarga todas (8500, tarda ~2 horas)
"""

import argparse
import io
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

# -- Configuracion --------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

HF_DATASET = "neuralcatcher/hateful_memes"
HF_REPO_URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

DEFAULT_MAX = 1500  # suficiente para entrenar y evaluar


def download_dataset(max_samples: int = DEFAULT_MAX):
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
    total = len(ds)
    limit = min(max_samples, total) if max_samples > 0 else total
    print(f"  Total en HuggingFace: {total}")
    print(f"  Limite de descarga:   {limit}")

    records = []
    downloaded = 0
    skipped = 0
    failed = 0

    for i, item in enumerate(tqdm(ds, desc="Descargando imagenes", total=total)):
        # Verificar si ya alcanzamos el limite de imagenes procesadas
        if len(records) >= limit:
            break

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
        except Exception:
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
    parser = argparse.ArgumentParser(
        description="Descarga el dataset Hateful Memes desde HuggingFace."
    )
    parser.add_argument(
        "--max", type=int, default=DEFAULT_MAX,
        help=f"Numero maximo de muestras a descargar (default: {DEFAULT_MAX}, 0 = todas)"
    )
    args = parser.parse_args()

    csv_path = DATA_DIR / "data.csv"
    n_images = len(list(IMG_DIR.glob("*.png")))
    if csv_path.exists() and n_images >= args.max and args.max > 0:
        print(f"El dataset ya existe ({n_images} imagenes).")
        print("Para re-descargar, elimina data/data.csv")
    else:
        download_dataset(max_samples=args.max)
