"""
data_loader.py — Carga del dataset y análisis exploratorio (EDA).
"""

import os

import pandas as pd

from .config import DATA_DIR, IMG_DIR


def load_dataset() -> pd.DataFrame:
    """
    Lee data.csv y reconstruye rutas de imágenes de forma portátil.
    Devuelve solo filas cuya imagen exista en disco.
    """
    csv_path = DATA_DIR / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}")

    df = pd.read_csv(csv_path)

    # Reconstruir rutas portátiles a partir del original_id
    df["img_path"] = df["original_id"].apply(
        lambda oid: str(IMG_DIR / f"{oid:05d}.png")
        if isinstance(oid, (int, float))
        else str(IMG_DIR / f"{oid}.png")
    )

    # Filtrar muestras con imagen existente
    df = df[df["img_path"].apply(os.path.exists)].reset_index(drop=True)
    df["id"] = range(len(df))

    return df


def add_eda_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas auxiliares para EDA."""
    df = df.copy()
    df["text_length"] = df["text"].astype(str).apply(len)
    df["clase"] = df["label"].map({0: "No ofensivo", 1: "Ofensivo"})
    return df


def print_summary(df: pd.DataFrame) -> None:
    """Imprime resumen del dataset en consola."""
    print(f"Dataset cargado: {len(df)} memes con imágenes verificadas")
    print(f"\nDistribución de clases:")
    print(
        df["label"]
        .value_counts()
        .rename({0: "No ofensivo (0)", 1: "Ofensivo (1)"})
        .to_string()
    )
