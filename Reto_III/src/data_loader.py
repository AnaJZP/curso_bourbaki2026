"""
data_loader.py -- Carga y preprocesamiento de datos del Reto III.

Limpia el CSV de listings de Airbnb Santorini:
- Parsea precios
- Imputa NaN
- One-hot encoding de variables categoricas
- Devuelve DataFrame limpio y matriz de features
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, LISTING_FILE, NUMERIC_FEATURES, TEXT_FEATURES


def load_listings() -> pd.DataFrame:
    """Carga el CSV crudo de listings."""
    path = DATA_DIR / LISTING_FILE
    df = pd.read_csv(path)
    return df


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia la columna price (ya es int en este dataset)."""
    df = df.copy()

    # Si price es string con $, limpiar
    if df["price"].dtype == object:
        df["price"] = (
            df["price"]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )

    # Filtrar precios invalidos
    df = df[df["price"] > 0].copy()
    return df


def extract_bathrooms(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae numero de banos de bathrooms_text."""
    df = df.copy()
    if "bathrooms_text" in df.columns:
        df["bathrooms_num"] = (
            df["bathrooms_text"]
            .astype(str)
            .str.extract(r"(\d+\.?\d*)")
            .astype(float)
        )
        df["bathrooms_num"] = df["bathrooms_num"].fillna(1.0)
    else:
        df["bathrooms_num"] = 1.0
    return df


def preprocess(df: pd.DataFrame, log_price: bool = True) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Pipeline completo de preprocesamiento.

    Retorna:
        df_clean: DataFrame limpio con todas las columnas
        X: matriz de features (N, F) estandarizada
        y: vector de precios (N,)
        feature_names: nombres de las features
    """
    df = clean_price(df)
    df = extract_bathrooms(df)

    # Filtrar outliers de precio (percentil 99)
    p99 = df["price"].quantile(0.99)
    df = df[df["price"] <= p99].copy()

    # One-hot de room_type
    room_dummies = pd.get_dummies(df["room_type"], prefix="room", dtype=float)
    df = pd.concat([df, room_dummies], axis=1)

    # Superhost como binario
    df["is_superhost"] = (df["host_is_superhost"] == "t").astype(float)

    # Instant bookable como binario
    df["is_instant"] = (df["instant_bookable"] == "t").astype(float)

    # Lista final de features
    feature_cols = NUMERIC_FEATURES.copy()
    feature_cols.append("bathrooms_num")
    feature_cols.append("is_superhost")
    feature_cols.append("is_instant")
    feature_cols.extend(room_dummies.columns.tolist())

    # Filtrar features que existen
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Imputar NaN con mediana
    for c in feature_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    X = df[feature_cols].values.astype(np.float32)
    y = df["price"].values.astype(np.float32)

    # Transformar precio con log1p para manejar asimetria
    if log_price:
        y = np.log1p(y).astype(np.float32)

    # Estandarizar
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Reset index
    df = df.reset_index(drop=True)

    return df, X.astype(np.float32), y, feature_cols


def build_text_corpus(df: pd.DataFrame) -> list[str]:
    """Construye un corpus de texto por listing concatenando name + description."""
    texts = []
    for _, row in df.iterrows():
        parts = []
        for col in TEXT_FEATURES:
            val = row.get(col, "")
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        texts.append(" ".join(parts) if parts else "listing")
    return texts


def print_summary(df: pd.DataFrame):
    """Imprime un resumen del dataset."""
    print(f"  Listings:      {len(df):,}")
    print(f"  Columnas:      {df.shape[1]}")
    print(f"  Precio medio:  ${df['price'].mean():.0f}")
    print(f"  Precio mediana:${df['price'].median():.0f}")
    print(f"  Rango:         ${df['price'].min():.0f} - ${df['price'].max():.0f}")
    print(f"  Room types:    {df['room_type'].value_counts().to_dict()}")
    print(f"  Con lat/lon:   {df[['latitude','longitude']].notna().all(axis=1).sum():,}")
