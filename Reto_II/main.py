#!/usr/bin/env python3
"""
main.py -- Punto de entrada del Reto II: Series Temporales.

Ejecutar:
    python main.py

Genera graficas y tabla resumen en la carpeta resultados/.
"""

import sys

# Corregir encoding en Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import DEVICE, OUTPUT_DIR
from src.data_loader import (
    build_asset_week_features,
    load_raw_data,
    print_summary,
)
from src.features import build_feature_matrix
from src.training import run_all_models
from src.visualization import (
    build_summary_table,
    plot_capacity_distribution,
    plot_feature_importance,
    plot_lstm_learning,
    plot_measure_distributions,
    plot_mse_comparison,
    plot_nan_heatmap,
    plot_predictions,
    plot_production_series,
)


def main():
    print("=" * 60)
    print("  RETO II -- Series Temporales: Produccion Industrial")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    # 1. Carga de datos
    print("\n[1/6] Cargando datos ...")
    X_raw, y, assets = load_raw_data()
    print_summary(X_raw, y, assets)

    # 2. EDA y visualizaciones
    print("\n[2/6] Generando graficas de EDA ...")
    plot_production_series(y)
    plot_capacity_distribution(assets)
    plot_nan_heatmap(X_raw)
    plot_measure_distributions(X_raw)

    # 3. Feature engineering
    print("\n[3/6] Construyendo features ...")
    pivot = build_asset_week_features(X_raw, assets)
    print(f"  Tabla pivoteada: {pivot.shape[0]} filas "
          f"(activo-semana), {pivot.shape[1]} columnas")
    group_features = build_feature_matrix(pivot, y)
    for g, (X_g, y_g, ids) in group_features.items():
        print(f"  Grupo {g}: {X_g.shape[1]} features, {len(ids)} semanas")

    # 4. Entrenamiento de modelos
    print("\n[4/6] Entrenando modelos ...")
    results = run_all_models(pivot, y, group_features)

    # 5. Visualizacion de resultados
    print("\n[5/6] Generando graficas de resultados ...")
    plot_predictions(results)
    plot_mse_comparison(results)
    plot_feature_importance(results)
    plot_lstm_learning(results)

    # 6. Tabla resumen
    print("\n[6/6] Resumen final:")
    summary = build_summary_table(results)
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("  EJECUCION COMPLETADA")
    print(f"  Resultados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
