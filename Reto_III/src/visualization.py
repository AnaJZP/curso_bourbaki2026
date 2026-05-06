"""
visualization.py -- Graficas del Reto III.
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
    COLOR_ACCENT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLORS,
    OUTPUT_DIR,
)


def _save(fig, filename: str) -> str:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")
    return str(path)


# -- EDA ----------------------------------------------------------

def plot_price_distribution(df: pd.DataFrame) -> str:
    """Histograma de distribucion de precios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histograma
    axes[0].hist(df["price"], bins=50, color=COLOR_PRIMARY,
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(df["price"].mean(), color=COLOR_SECONDARY,
                    ls="--", lw=2, label=f"Media: ${df['price'].mean():.0f}")
    axes[0].axvline(df["price"].median(), color=COLOR_ACCENT,
                    ls=":", lw=2, label=f"Mediana: ${df['price'].median():.0f}")
    axes[0].set_title("Distribucion de Precios", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Precio (USD/noche)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend(fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    # Log-scale
    prices_log = np.log1p(df["price"])
    axes[1].hist(prices_log, bins=50, color=COLOR_SECONDARY,
                 edgecolor="white", alpha=0.85)
    axes[1].set_title("Distribucion de log(Precio + 1)",
                      fontsize=14, fontweight="bold")
    axes[1].set_xlabel("log(Precio + 1)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Precios de Airbnb en Santorini",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "01_distribucion_precios.png")


def plot_property_map(df: pd.DataFrame) -> str:
    """Scatter plot de propiedades coloreadas por precio."""
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(
        df["longitude"], df["latitude"],
        c=df["price"], cmap="Blues",
        s=15, alpha=0.6, edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Precio (USD/noche)", fontsize=12)

    ax.set_title("Mapa de Propiedades -- Santorini",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitud", fontsize=12)
    ax.set_ylabel("Latitud", fontsize=12)
    ax.grid(alpha=0.2)

    return _save(fig, "02_mapa_propiedades.png")


def plot_room_type(df: pd.DataFrame) -> str:
    """Distribucion por tipo de habitacion y boxplot de precios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    shades = ["#1B4F72", "#2471A3", "#2E86C1", "#85C1E9"]

    # Conteo
    counts = df["room_type"].value_counts()
    bars = axes[0].bar(range(len(counts)), counts.values,
                       color=shades[:len(counts)], edgecolor="white")
    axes[0].set_xticks(range(len(counts)))
    axes[0].set_xticklabels(counts.index, rotation=30, ha="right", fontsize=10)
    axes[0].set_title("Listings por Tipo de Habitacion",
                      fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Cantidad")
    axes[0].grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     str(val), ha="center", va="bottom", fontsize=11,
                     fontweight="bold")

    # Boxplot
    room_types = sorted(df["room_type"].unique())
    data_bp = [df[df["room_type"] == rt]["price"].values for rt in room_types]
    bp = axes[1].boxplot(data_bp, labels=room_types, patch_artist=True,
                         showfliers=False)
    for patch, color in zip(bp["boxes"], shades[:len(room_types)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_title("Precio por Tipo de Habitacion",
                      fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Precio (USD/noche)")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle("Analisis por Tipo de Habitacion",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "03_tipo_habitacion.png")


def plot_graph_sample(data, title: str, filename: str,
                      df: pd.DataFrame, max_nodes: int = 300) -> str:
    """Visualizacion de un subgrafo (submuestra de nodos)."""
    fig, ax = plt.subplots(figsize=(12, 8))

    n = min(max_nodes, data.num_nodes)
    node_subset = set(range(n))

    lons = df["longitude"].values[:n]
    lats = df["latitude"].values[:n]
    prices = df["price"].values[:n]

    # Dibujar aristas
    edge_index = data.edge_index.cpu().numpy()
    drawn = 0
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        if src < n and dst < n:
            ax.plot([lons[src], lons[dst]], [lats[src], lats[dst]],
                    color=COLOR_ACCENT, alpha=0.08, lw=0.5)
            drawn += 1

    # Dibujar nodos
    scatter = ax.scatter(lons, lats, c=prices, cmap="Blues",
                         s=20, alpha=0.8, edgecolors="white", linewidths=0.3,
                         zorder=5)
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label("Precio (USD/noche)", fontsize=11)

    ax.set_title(f"{title}\n({n} nodos, {drawn} aristas visibles)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitud", fontsize=12)
    ax.set_ylabel("Latitud", fontsize=12)
    ax.grid(alpha=0.2)

    return _save(fig, filename)


# -- Resultados ---------------------------------------------------

def plot_predictions(results: dict) -> str:
    """Real vs predicho para cada modelo."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    model_colors = [COLOR_SECONDARY, COLOR_PRIMARY, COLOR_ACCENT]

    for ax, (name, res), color in zip(axes, results.items(), model_colors):
        ax.scatter(res.y_true, res.y_pred, alpha=0.3, s=10,
                   color=color, edgecolors="none")

        # Linea perfecta
        lims = [
            min(res.y_true.min(), res.y_pred.min()),
            max(res.y_true.max(), res.y_pred.max()),
        ]
        ax.plot(lims, lims, "--", color="#333333", alpha=0.5, lw=1.5)

        ax.set_title(f"{name.upper()}\nR2={res.r2:.4f}, MAE={res.mae:.1f}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Precio Real (USD)")
        ax.set_ylabel("Precio Predicho (USD)")
        ax.grid(alpha=0.3)

    plt.suptitle("Precio Real vs Predicho",
                 fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    return _save(fig, "05_predicciones.png")


def plot_metrics_comparison(results: dict) -> str:
    """Barplot comparativo de metricas."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    model_names = list(results.keys())
    colors_list = [COLORS.get(n, COLOR_PRIMARY) for n in model_names]
    x = np.arange(len(model_names))

    # MSE
    mses = [results[n].mse for n in model_names]
    bars = axes[0].bar(x, mses, color=colors_list, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.upper() for n in model_names], fontsize=11)
    axes[0].set_title("MSE", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mses):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    # MAE
    maes = [results[n].mae for n in model_names]
    bars = axes[1].bar(x, maes, color=colors_list, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.upper() for n in model_names], fontsize=11)
    axes[1].set_title("MAE", fontsize=14, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, maes):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    # R2
    r2s = [results[n].r2 for n in model_names]
    bars = axes[2].bar(x, r2s, color=colors_list, edgecolor="white")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([n.upper() for n in model_names], fontsize=11)
    axes[2].set_title("R2 Score", fontsize=14, fontweight="bold")
    axes[2].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, r2s):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Comparacion de Metricas por Modelo",
                 fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    return _save(fig, "06_comparacion_metricas.png")


def plot_learning_curves(results: dict) -> str:
    """Curvas de aprendizaje de los modelos GCN."""
    gcn_results = {k: v for k, v in results.items() if v.history}
    if not gcn_results:
        return ""

    n = len(gcn_results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    colors_iter = [COLOR_PRIMARY, COLOR_ACCENT]

    for ax, (name, res), c in zip(axes, gcn_results.items(), colors_iter):
        h = res.history
        ax.plot(h["train_loss"], label="Train", color=c, lw=2)
        ax.plot(h["val_loss"], label="Val", color=c, lw=2, ls="--")
        ax.set_title(f"Curva de Aprendizaje -- {name.upper()}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle("Curvas de Aprendizaje GCN",
                 fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    return _save(fig, "07_learning_curves.png")


def build_summary_table(results: dict) -> pd.DataFrame:
    """Tabla resumen de resultados."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Modelo": name.upper(),
            "MSE": f"{res.mse:,.2f}",
            "RMSE": f"{res.rmse:,.2f}",
            "MAE": f"{res.mae:,.2f}",
            "R2": f"{res.r2:.4f}",
        })

    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "08_resumen_resultados.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Resumen guardado: {csv_path}")
    return summary
