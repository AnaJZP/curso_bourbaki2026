"""
visualization.py -- Graficas del Reto II.
Paleta unificada: escala de azules.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    CMAP_HEATMAP,
    COLOR_GROUP_2,
    COLOR_GROUP_3,
    COLORS,
    GROUPS,
    OUTPUT_DIR,
)


def _save(fig, filename: str) -> str:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")
    return str(path)


# ── EDA ──────────────────────────────────────────────────────

def plot_production_series(y: pd.DataFrame) -> str:
    """Serie temporal de produccion por grupo."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    colors = {2: COLOR_GROUP_2, 3: COLOR_GROUP_3}

    for ax, g in zip(axes, GROUPS):
        col = f"PRODUCTION_GROUP_{g}"
        ax.plot(y["SAMPLE_ID"], y[col], color=colors[g], lw=2)
        ax.fill_between(y["SAMPLE_ID"], y[col], alpha=0.15, color=colors[g])
        ax.set_title(f"Produccion Semanal -- Grupo {g}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Semana (SAMPLE_ID)")
        ax.set_ylabel("Produccion")
        ax.grid(alpha=0.3)

    plt.suptitle("Series de Produccion Semanal por Grupo",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "01_produccion_semanal.png")


def plot_capacity_distribution(assets: pd.DataFrame) -> str:
    """Distribucion de capacidades nominales."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(assets["ASSET_NOMINAL_CAPACITY"], bins=25,
            color=COLOR_GROUP_2, edgecolor="white", alpha=0.85)
    ax.axvline(assets["ASSET_NOMINAL_CAPACITY"].mean(), color=COLOR_GROUP_3,
               ls="--", lw=2, label=f"Media: {assets['ASSET_NOMINAL_CAPACITY'].mean():,.0f}")
    ax.set_title("Distribucion de Capacidades Nominales (83 activos)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Capacidad Nominal")
    ax.set_ylabel("Frecuencia")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "02_capacidades_nominales.png")


def plot_nan_heatmap(X_raw: pd.DataFrame) -> str:
    """Heatmap de NaN por activo y tipo de medida."""
    nan_pct = X_raw.groupby(["ASSET_ID", "MEASURE_TYPE"])["MEASURE_VALUE"].apply(
        lambda x: x.isna().mean()
    ).unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 14))
    sns.heatmap(nan_pct, cmap=CMAP_HEATMAP, ax=ax, vmin=0, vmax=1,
                cbar_kws={"label": "Fraccion de NaN"})
    ax.set_title("Valores Faltantes por Activo y Tipo de Medida",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Tipo de Medida")
    ax.set_ylabel("Activo (ASSET_ID)")
    return _save(fig, "03_nan_heatmap.png")


def plot_measure_distributions(X_raw: pd.DataFrame) -> str:
    """Distribucion de cada tipo de medida."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    shades = ["#1B4F72", "#2471A3", "#2E86C1", "#85C1E9"]

    for ax, mt, c in zip(axes, sorted(X_raw["MEASURE_TYPE"].unique()), shades):
        data = X_raw[X_raw["MEASURE_TYPE"] == mt]["MEASURE_VALUE"].dropna()
        ax.hist(data, bins=50, color=c, edgecolor="white", alpha=0.85)
        ax.set_title(f"Medida Tipo {mt}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frecuencia")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Distribucion de Mediciones por Tipo",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "04_distribucion_medidas.png")


# ── Resultados ───────────────────────────────────────────────

def plot_predictions(results: dict) -> str:
    """Produccion real vs predicha para cada modelo y grupo."""
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 5 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    colors_g = {2: COLOR_GROUP_2, 3: COLOR_GROUP_3}

    for row, (model_name, res) in enumerate(results.items()):
        for col, g in enumerate(GROUPS):
            ax = axes[row, col]
            trues = res.true_per_group[g]
            preds = res.preds_per_group[g]
            weeks = range(len(trues))

            ax.plot(weeks, trues, color=colors_g[g], lw=2, label="Real")
            ax.plot(weeks, preds, color=colors_g[g], lw=2, ls="--",
                    alpha=0.7, label="Predicho")
            ax.fill_between(weeks, trues, preds, alpha=0.1, color=colors_g[g])
            ax.set_title(f"{model_name.upper()} -- Grupo {g} "
                         f"(MSE={res.mse_per_group[g]:.2e})",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Semana (validacion)")
            ax.set_ylabel("Produccion")
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

    plt.suptitle("Produccion Real vs Predicha",
                 fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return _save(fig, "05_predicciones.png")


def plot_mse_comparison(results: dict) -> str:
    """Comparacion de MSE entre modelos por grupo."""
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = list(results.keys())
    x = np.arange(len(GROUPS))
    width = 0.25

    for i, name in enumerate(model_names):
        mses = [results[name].mse_per_group[g] for g in GROUPS]
        bars = ax.bar(x + i * width, mses, width, label=name.upper(),
                      color=COLORS[name], edgecolor="white")
        for bar, mse in zip(bars, mses):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{mse:.2e}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Grupo {g}" for g in GROUPS], fontsize=13)
    ax.set_ylabel("MSE", fontsize=13)
    ax.set_title("Comparacion de MSE por Modelo y Grupo",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "06_comparacion_mse.png")


def plot_feature_importance(results: dict) -> str:
    """Top 15 features mas importantes del GBR por grupo."""
    if "gbr" not in results or not results["gbr"].feature_importance:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    shades = [COLOR_GROUP_2, COLOR_GROUP_3]

    for ax, g, c in zip(axes, GROUPS, shades):
        fi = results["gbr"].feature_importance.get(g, {})
        if not fi:
            continue
        fi_sorted = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]
        names = [x[0] for x in fi_sorted]
        vals = [x[1] for x in fi_sorted]

        ax.barh(names[::-1], vals[::-1], color=c, edgecolor="white")
        ax.set_title(f"Feature Importance -- Grupo {g}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Importancia")
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle("Top 15 Features (GradientBoosting)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "07_feature_importance.png")


def plot_lstm_learning(results: dict) -> str:
    """Curvas de aprendizaje del LSTM."""
    if "lstm" not in results or not results["lstm"].history:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    shades = [COLOR_GROUP_2, COLOR_GROUP_3]

    for ax, g, c in zip(axes, GROUPS, shades):
        h = results["lstm"].history.get(g, {})
        if not h:
            continue
        ax.plot(h["train_loss"], label="Train", color=c, lw=2)
        ax.plot(h["val_loss"], label="Val", color=c, lw=2, ls="--")
        ax.set_title(f"LSTM Curva de Aprendizaje -- Grupo {g}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE (normalizado)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Curvas de Aprendizaje LSTM",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "08_lstm_learning.png")


def build_summary_table(results: dict) -> pd.DataFrame:
    """Tabla resumen de resultados."""
    rows = []
    for name, res in results.items():
        for g in GROUPS:
            rows.append({
                "Modelo": name.upper(),
                "Grupo": g,
                "MSE": f"{res.mse_per_group[g]:.4e}",
                "RMSE": f"{np.sqrt(res.mse_per_group[g]):,.0f}",
            })

    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "09_resumen_resultados.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Resumen guardado: {csv_path}")
    return summary
