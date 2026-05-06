"""
features.py -- Feature engineering para el Reto II.

Genera la matriz de features lista para entrenar modelos.
"""

import numpy as np
import pandas as pd

from .config import GROUPS


def build_feature_matrix(
    pivot: pd.DataFrame,
    y: pd.DataFrame,
) -> dict[int, tuple[pd.DataFrame, np.ndarray, np.ndarray]]:
    """
    Construye la matriz de features y targets por grupo.

    Para cada grupo k:
    1. Filtra activos del grupo
    2. Para cada SAMPLE_ID (semana), calcula la produccion estimada como
       sum(capacity_factor_i * nominal_capacity_i) para cada activo i
    3. El target es PRODUCTION_GROUP_{k}

    Retorna: {group_id: (X_features_por_semana, y_target, sample_ids)}
    """
    result = {}

    for g in GROUPS:
        group_data = pivot[pivot["GROUP_ID"] == g].copy()
        feature_cols = [c for c in group_data.columns
                        if c.startswith("mt") or c == "ASSET_NOMINAL_CAPACITY"]

        # Agrupar por SAMPLE_ID: agregar features de todos los activos del grupo
        # Estrategia: por cada semana, concatenar estadisticas de todos los activos
        weeks = sorted(group_data["SAMPLE_ID"].unique())
        X_rows = []
        sample_ids = []

        for week in weeks:
            week_data = group_data[group_data["SAMPLE_ID"] == week]

            # Features agregadas del grupo para esta semana
            row = {}

            # Estadisticas globales por tipo de medida
            for mt in range(1, 5):
                mt_cols = [c for c in feature_cols if c.startswith(f"mt{mt}_")]
                if mt_cols:
                    vals = week_data[mt_cols].values.flatten()
                    vals = vals[~np.isnan(vals)]
                    row[f"g{g}_mt{mt}_mean"] = np.mean(vals) if len(vals) > 0 else 0
                    row[f"g{g}_mt{mt}_std"]  = np.std(vals) if len(vals) > 0 else 0
                    row[f"g{g}_mt{mt}_max"]  = np.max(vals) if len(vals) > 0 else 0
                    row[f"g{g}_mt{mt}_min"]  = np.min(vals) if len(vals) > 0 else 0
                    row[f"g{g}_mt{mt}_sum"]  = np.sum(vals)

            # Capacidad total del grupo
            row[f"g{g}_total_capacity"] = week_data["ASSET_NOMINAL_CAPACITY"].sum()

            # Numero de activos con alguna medida no-cero (proxy de "activos")
            for mt in range(1, 5):
                mt_cols_raw = [c for c in week_data.columns if c.startswith(f"mt{mt}_wd")]
                if mt_cols_raw:
                    n_active = (week_data[mt_cols_raw].abs().sum(axis=1) > 0).sum()
                    row[f"g{g}_mt{mt}_n_active"] = n_active

            X_rows.append(row)
            sample_ids.append(week)

        X_df = pd.DataFrame(X_rows)
        target_col = f"PRODUCTION_GROUP_{g}"
        y_vals = y.set_index("SAMPLE_ID").loc[sample_ids, target_col].values

        result[g] = (X_df, y_vals, np.array(sample_ids))

    return result


def build_asset_level_features(
    pivot: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Construye features a nivel de activo individual para predecir
    factor de capacidad (produccion / capacidad_nominal).

    Retorna: (X_features, asset_info_df con SAMPLE_ID, ASSET_ID, GROUP_ID, capacity)
    """
    feature_cols = [c for c in pivot.columns if c.startswith("mt")]

    X = pivot[feature_cols].copy()
    info = pivot[["SAMPLE_ID", "ASSET_ID", "GROUP_ID", "ASSET_NOMINAL_CAPACITY"]].copy()

    return X, info
