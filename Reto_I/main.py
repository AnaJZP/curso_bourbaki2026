#!/usr/bin/env python3
"""
main.py -- Punto de entrada del Reto I: Redes Multi-Modales.

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
from src.data_loader import add_eda_columns, load_dataset, print_summary
from src.embeddings import extract_embeddings, load_clip
from src.fusion import fuse_all
from src.training import run_experiments
from src.visualization import (
    build_summary_table,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_learning_curves,
    plot_roc_curves,
    plot_samples,
    plot_text_length,
)


def main():
    print("=" * 60)
    print("  RETO I -- Redes Multi-Modales: Deteccion de Memes Ofensivos")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    # 1. Carga de datos
    print("\n[1/7] Cargando dataset ...")
    df = load_dataset()
    print_summary(df)
    df = add_eda_columns(df)

    # 2. Visualizacion exploratoria
    print("\n[2/7] Generando graficas de EDA ...")
    plot_samples(df)
    plot_class_distribution(df)
    plot_text_length(df)

    # 3. Embeddings CLIP
    print("\n[3/7] Extrayendo embeddings con CLIP ...")
    clip_model, clip_processor = load_clip()
    text_emb, image_emb = extract_embeddings(df, clip_processor, clip_model)
    print(f"  Text embeddings:  {text_emb.shape}")
    print(f"  Image embeddings: {image_emb.shape}")

    # 4. Fusion multi-modal
    print("\n[4/7] Aplicando estrategias de fusion ...")
    fusions = fuse_all(text_emb, image_emb)
    for name, X in fusions.items():
        print(f"  Fusion '{name}': shape = {X.shape}")

    # 5. Entrenamiento
    print("\n[5/7] Entrenando clasificadores ...")
    labels = df["label"].values
    results = run_experiments(fusions, labels)

    # 6. Visualizacion de resultados
    print("\n[6/7] Generando graficas de resultados ...")
    plot_learning_curves(results)
    plot_roc_curves(results)
    plot_confusion_matrices(results)

    # 7. Tabla resumen
    print("\n[7/7] Resumen final:")
    summary = build_summary_table(results, fusions)
    print(summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("  EJECUCION COMPLETADA")
    print(f"  Resultados en: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
