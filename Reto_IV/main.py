#!/usr/bin/env python3
"""
main.py -- Punto de entrada del Reto IV: Deep Reinforcement Learning.

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
from src.data_loader import load_data, preprocess, print_summary
from src.environment import DynamicPricingEnv
from src.training import run_all_models
from src.visualization import (
    build_summary_table,
    plot_actions_heatmap,
    plot_demand_supply,
    plot_epsilon_decay,
    plot_multiplier_distribution,
    plot_price_distribution,
    plot_revenue_comparison,
    plot_reward_curve,
)


def main():
    print("=" * 60)
    print("  RETO IV -- Deep Reinforcement Learning: Dynamic Pricing")
    print(f"  Dispositivo: {DEVICE}")
    print("=" * 60)

    # 1. Carga de datos
    print("\n[1/7] Cargando dataset ...")
    df = load_data()
    print_summary(df)

    # 2. Preprocesamiento
    print("\n[2/7] Preprocesando datos ...")
    df_proc, states, base_prices = preprocess(df)
    print(f"  Estados: {states.shape} (features x registros)")
    print(f"  Precios base: rango ${base_prices.min():.2f} - ${base_prices.max():.2f}")

    # 3. Visualizacion exploratoria
    print("\n[3/7] Generando graficas de EDA ...")
    plot_demand_supply(df)
    plot_price_distribution(df)

    # 4. Crear entorno MDP
    print("\n[4/7] Creando entorno MDP ...")
    demand_ratios = (df["Number_of_Riders"] / df["Number_of_Drivers"].clip(lower=1)).values
    env = DynamicPricingEnv(states, base_prices, demand_ratios)
    print(f"  State dim:  {env.state_dim}")
    print(f"  N acciones: {env.n_actions}")
    print(f"  N muestras: {env.n_samples}")

    # 5. Entrenamiento y evaluacion
    print("\n[5/7] Entrenando y evaluando modelos ...")
    results, agent, history = run_all_models(env)

    # 6. Visualizacion de resultados
    print("\n[6/7] Generando graficas de resultados ...")
    plot_reward_curve(history)
    plot_epsilon_decay(history)
    plot_actions_heatmap(results, env)
    plot_revenue_comparison(results)
    plot_multiplier_distribution(results)

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
