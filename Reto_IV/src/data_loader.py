"""
data_loader.py -- Carga y preprocesamiento de datos del Reto IV.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DATA_DIR, DATA_FILE


# Columnas categoricas y su orden
CATEGORICAL_COLS = {
    "Location_Category":      ["Rural", "Suburban", "Urban"],
    "Customer_Loyalty_Status": ["Regular", "Silver", "Gold"],
    "Time_of_Booking":        ["Morning", "Afternoon", "Evening", "Night"],
    "Vehicle_Type":           ["Economy", "Premium"],
}


def load_data() -> pd.DataFrame:
    """Carga el CSV crudo de dynamic pricing."""
    path = DATA_DIR / DATA_FILE
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Preprocesa el dataset.

    Retorna:
        df: DataFrame con columnas adicionales
        states: matriz de estados (N, state_dim) normalizada
        base_prices: vector de precios base (Historical_Cost_of_Ride)
    """
    df = df.copy()

    # Ratio demanda/oferta (feature clave para pricing)
    df["demand_supply_ratio"] = df["Number_of_Riders"] / df["Number_of_Drivers"].clip(lower=1)

    # Encoding ordinal de categoricas
    encoders = {}
    for col, categories in CATEGORICAL_COLS.items():
        le = LabelEncoder()
        le.classes_ = np.array(categories)
        # Mapear valores conocidos, dejar -1 para desconocidos
        df[f"{col}_enc"] = df[col].apply(
            lambda x: list(le.classes_).index(x) if x in le.classes_ else 0
        )
        encoders[col] = le

    # Features del estado
    state_cols = [
        "demand_supply_ratio",
        "Number_of_Riders",
        "Number_of_Drivers",
        "Number_of_Past_Rides",
        "Average_Ratings",
        "Expected_Ride_Duration",
        "Location_Category_enc",
        "Customer_Loyalty_Status_enc",
        "Time_of_Booking_enc",
        "Vehicle_Type_enc",
    ]

    states = df[state_cols].values.astype(np.float32)

    # Normalizar
    scaler = StandardScaler()
    states = scaler.fit_transform(states).astype(np.float32)

    base_prices = df["Historical_Cost_of_Ride"].values.astype(np.float32)

    return df, states, base_prices


def print_summary(df: pd.DataFrame):
    """Imprime un resumen del dataset."""
    print(f"  Registros:        {len(df):,}")
    print(f"  Columnas:         {df.shape[1]}")
    print(f"  Precio medio:     ${df['Historical_Cost_of_Ride'].mean():.2f}")
    print(f"  Precio rango:     ${df['Historical_Cost_of_Ride'].min():.2f}"
          f" - ${df['Historical_Cost_of_Ride'].max():.2f}")
    print(f"  Riders:           {df['Number_of_Riders'].min()}"
          f"-{df['Number_of_Riders'].max()}")
    print(f"  Drivers:          {df['Number_of_Drivers'].min()}"
          f"-{df['Number_of_Drivers'].max()}")
    print(f"  Ubicaciones:      {df['Location_Category'].value_counts().to_dict()}")
    print(f"  Vehiculos:        {df['Vehicle_Type'].value_counts().to_dict()}")
