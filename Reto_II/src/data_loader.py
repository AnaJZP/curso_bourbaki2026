"""
data_loader.py -- Carga y transformacion de datos del Reto II.

Transforma el formato tidy (241K filas) a un formato plano donde cada fila
es un activo-semana con sus features de medicion.
"""

import pandas as pd
import numpy as np

from .config import DATA_DIR, GROUPS


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los 3 CSVs crudos."""
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv")
    assets = pd.read_csv(DATA_DIR / "assets.csv")

    # Eliminar columna sin nombre
    if "Unnamed: 0" in X.columns:
        X = X.drop(columns=["Unnamed: 0"])

    return X, y, assets


def print_summary(X: pd.DataFrame, y: pd.DataFrame, assets: pd.DataFrame):
    """Imprime un resumen del dataset."""
    print(f"  X_train:  {X.shape[0]:,} registros, {X.shape[1]} columnas")
    print(f"  y_train:  {y.shape[0]} semanas, {y.shape[1] - 1} grupos")
    print(f"  assets:   {assets.shape[0]} activos")
    print(f"  Grupos:   {sorted(X['GROUP_ID'].unique())}")
    print(f"  NaN en MEASURE_VALUE: {X['MEASURE_VALUE'].isna().sum():,} "
          f"({X['MEASURE_VALUE'].isna().mean():.1%})")


def build_asset_week_features(X: pd.DataFrame, assets: pd.DataFrame) -> pd.DataFrame:
    """
    Pivotea el dataframe tidy a formato plano: una fila por (SAMPLE_ID, ASSET_ID).

    Features generadas por fila:
    - mt{j}_wd{l}: valor de medida tipo j en dia l (4x7 = 28 features)
    - mt{j}_mean, mt{j}_std, mt{j}_max, mt{j}_min: agregados por tipo (4x4 = 16)
    - ASSET_NOMINAL_CAPACITY: capacidad nominal del activo
    - GROUP_ID: grupo al que pertenece
    """
    # Crear columna de feature name
    X = X.copy()
    X["feat_name"] = "mt" + X["MEASURE_TYPE"].astype(str) + "_wd" + X["MEASURE_WEEKDAY"].astype(str)

    # Pivot: cada fila = (SAMPLE_ID, ASSET_ID), columnas = feat_name
    pivot = X.pivot_table(
        index=["SAMPLE_ID", "GROUP_ID", "ASSET_ID"],
        columns="feat_name",
        values="MEASURE_VALUE",
        aggfunc="first",
    ).reset_index()

    # Aplanar multiindex de columnas
    pivot.columns.name = None

    # Agregar estadisticas por tipo de medida
    for mt in sorted(X["MEASURE_TYPE"].unique()):
        cols_mt = [c for c in pivot.columns if c.startswith(f"mt{mt}_wd")]
        if cols_mt:
            pivot[f"mt{mt}_mean"] = pivot[cols_mt].mean(axis=1)
            pivot[f"mt{mt}_std"]  = pivot[cols_mt].std(axis=1)
            pivot[f"mt{mt}_max"]  = pivot[cols_mt].max(axis=1)
            pivot[f"mt{mt}_min"]  = pivot[cols_mt].min(axis=1)

    # Merge con capacidades nominales
    pivot = pivot.merge(assets, on="ASSET_ID", how="left")

    # Imputar NaN con 0 (medida no disponible = sin actividad anormal)
    feature_cols = [c for c in pivot.columns if c.startswith("mt")]
    pivot[feature_cols] = pivot[feature_cols].fillna(0)

    return pivot


def aggregate_targets(y: pd.DataFrame) -> pd.DataFrame:
    """Devuelve targets limpios."""
    return y.copy()


def get_group_assets(X: pd.DataFrame) -> dict[int, list[int]]:
    """Devuelve un dict {group_id: [lista de asset_ids]}."""
    result = {}
    for g in GROUPS:
        result[g] = sorted(X[X["GROUP_ID"] == g]["ASSET_ID"].unique().tolist())
    return result
