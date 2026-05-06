#!/usr/bin/env python3
"""
main.py -- Punto de entrada del Reto III: Graph Neural Networks.

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
    build_text_corpus,
    load_listings,
    preprocess,
    print_summary,
)
from src.graph_builder import build_geo_graph, build_text_graph
from src.training import run_all_models
from src.visualization import (
    build_summary_table,
    plot_graph_sample,
    plot_learning_curves,
    plot_metrics_comparison,
    plot_predictions,
    plot_price_distribution,
    plot_property_map,
    plot_room_type,
)


def main():
    print("=" * 60)
    print("  RETO III -- Graph Neural Networks: Precios de Airbnb")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    # 1. Carga de datos
    print("\n[1/7] Cargando dataset ...")
    df_raw = load_listings()
    print_summary(df_raw)

    # 2. Preprocesamiento
    print("\n[2/7] Preprocesando datos ...")
    df, X, y, feature_names = preprocess(df_raw)
    print(f"  Despues de limpieza: {len(df):,} listings")
    print(f"  Features: {len(feature_names)} columnas")
    print(f"  Target: log(precio + 1), rango [{y.min():.2f}, {y.max():.2f}]")

    # 3. Visualizacion exploratoria
    print("\n[3/7] Generando graficas de EDA ...")
    plot_price_distribution(df)
    plot_property_map(df)
    plot_room_type(df)

    # 4. Construccion de grafos
    print("\n[4/7] Construyendo grafos ...")
    geo_data = build_geo_graph(df, X, y)

    texts = build_text_corpus(df)
    txt_data = build_text_graph(texts, X, y)

    # 4b. Visualizar grafos
    plot_graph_sample(geo_data, "Grafo Geografico (KNN)",
                      "04a_grafo_geografico.png", df)
    plot_graph_sample(txt_data, "Grafo Textual (Similitud Coseno)",
                      "04b_grafo_textual.png", df)

    # 5. Entrenamiento
    print("\n[5/7] Entrenando modelos ...")
    results = run_all_models(X, y, geo_data, txt_data)

    # 6. Visualizacion de resultados
    print("\n[6/7] Generando graficas de resultados ...")
    plot_predictions(results)
    plot_metrics_comparison(results)
    plot_learning_curves(results)

    # 7. Tabla resumen
    print("\n[7/7] Resumen final:")
    summary = build_summary_table(results)
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("  EJECUCION COMPLETADA")
    print(f"  Resultados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
