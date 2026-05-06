"""
visualization.py -- Graficas del Reto IV.
Paleta unificada: escala de azules.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    CMAP_HEATMAP,
    COLOR_ACCENT,
    COLOR_LIGHT,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLORS,
    OUTPUT_DIR,
    PRICE_MULTIPLIERS,
)


def _save(fig, filename: str) -> str:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")
    return str(path)


# -- EDA ----------------------------------------------------------

def plot_demand_supply(df: pd.DataFrame) -> str:
    """Distribucion del ratio demanda/oferta."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Histograma de demand_supply_ratio
    ratio = df["Number_of_Riders"] / df["Number_of_Drivers"].clip(lower=1)
    axes[0].hist(ratio, bins=30, color=COLOR_PRIMARY,
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(1.0, color=COLOR_SECONDARY, ls="--", lw=2,
                    label="Equilibrio (1.0)")
    axes[0].set_title("Ratio Demanda / Oferta",
                      fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Riders / Drivers")
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend(fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    # Scatter riders vs drivers
    scatter = axes[1].scatter(
        df["Number_of_Drivers"], df["Number_of_Riders"],
        c=df["Historical_Cost_of_Ride"], cmap="Blues",
        s=20, alpha=0.6, edgecolors="none",
    )
    cbar = plt.colorbar(scatter, ax=axes[1], shrink=0.8)
    cbar.set_label("Precio Historico ($)", fontsize=11)
    axes[1].set_title("Riders vs Drivers",
                      fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Numero de Drivers")
    axes[1].set_ylabel("Numero de Riders")
    axes[1].plot([0, 100], [0, 100], "--", color="#333", alpha=0.3, lw=1)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Analisis de Demanda y Oferta",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "01_eda_demanda_oferta.png")


def plot_price_distribution(df: pd.DataFrame) -> str:
    """Distribucion de precios historicos por segmentos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    shades = ["#1B4F72", "#2471A3", "#2E86C1", "#85C1E9"]

    # Por tipo de vehiculo
    for i, vtype in enumerate(["Economy", "Premium"]):
        subset = df[df["Vehicle_Type"] == vtype]["Historical_Cost_of_Ride"]
        axes[0].hist(subset, bins=25, alpha=0.7, color=shades[i],
                     label=vtype, edgecolor="white")
    axes[0].set_title("Precio por Vehiculo", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Precio ($)")
    axes[0].legend(fontsize=11)
    axes[0].grid(axis="y", alpha=0.3)

    # Por ubicacion
    for i, loc in enumerate(["Rural", "Suburban", "Urban"]):
        subset = df[df["Location_Category"] == loc]["Historical_Cost_of_Ride"]
        axes[1].hist(subset, bins=25, alpha=0.7, color=shades[i],
                     label=loc, edgecolor="white")
    axes[1].set_title("Precio por Ubicacion", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Precio ($)")
    axes[1].legend(fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)

    # Por horario
    for i, time in enumerate(["Morning", "Afternoon", "Evening", "Night"]):
        subset = df[df["Time_of_Booking"] == time]["Historical_Cost_of_Ride"]
        axes[2].hist(subset, bins=25, alpha=0.65, color=shades[i],
                     label=time, edgecolor="white")
    axes[2].set_title("Precio por Horario", fontsize=14, fontweight="bold")
    axes[2].set_xlabel("Precio ($)")
    axes[2].legend(fontsize=11)
    axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Distribucion de Precios Historicos",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "02_eda_precios.png")


# -- Entrenamiento ------------------------------------------------

def plot_reward_curve(history: dict) -> str:
    """Curva de recompensa acumulada durante entrenamiento."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    rewards = history["episode_rewards"]

    # Recompensa por episodio (con media movil)
    axes[0].plot(rewards, alpha=0.3, color=COLOR_ACCENT, lw=0.8)
    window = 20
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(rewards)), moving_avg,
                     color=COLOR_PRIMARY, lw=2, label=f"Media movil ({window} ep)")
    axes[0].set_title("Recompensa por Episodio",
                      fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Episodio")
    axes[0].set_ylabel("Recompensa Total")
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Loss
    losses = history["episode_losses"]
    axes[1].plot(losses, color=COLOR_SECONDARY, lw=1, alpha=0.5)
    if len(losses) >= window:
        moving_avg_l = np.convolve(losses, np.ones(window)/window, mode="valid")
        axes[1].plot(range(window-1, len(losses)), moving_avg_l,
                     color=COLOR_SECONDARY, lw=2, label=f"Media movil ({window} ep)")
    axes[1].set_title("Loss del DQN",
                      fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Episodio")
    axes[1].set_ylabel("MSE Loss")
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.suptitle("Curvas de Entrenamiento DQN",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "03_reward_curve.png")


def plot_epsilon_decay(history: dict) -> str:
    """Decaimiento de epsilon."""
    fig, ax = plt.subplots(figsize=(10, 5))

    eps = history["epsilons"]
    ax.plot(eps, color=COLOR_PRIMARY, lw=2)
    ax.axhline(y=eps[-1], color=COLOR_SECONDARY, ls="--", lw=1.5,
               label=f"Epsilon final: {eps[-1]:.3f}")
    ax.fill_between(range(len(eps)), eps, alpha=0.15, color=COLOR_ACCENT)

    ax.set_title("Decaimiento de Epsilon (Exploracion)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Episodio")
    ax.set_ylabel("Epsilon")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    return _save(fig, "04_epsilon_decay.png")


def plot_actions_heatmap(results: dict, env) -> str:
    """Heatmap de acciones elegidas por el DQN vs demand_ratio."""
    fig, ax = plt.subplots(figsize=(12, 6))

    dqn_result = results.get("dqn")
    if not dqn_result:
        return ""

    demand_ratios = env.demand_ratios
    multipliers = dqn_result.multipliers_used

    # Bins para demand_ratio
    dr_bins = np.linspace(demand_ratios.min(), demand_ratios.max(), 11)
    mult_labels = [str(m) for m in PRICE_MULTIPLIERS]

    heatmap_data = np.zeros((len(PRICE_MULTIPLIERS), len(dr_bins) - 1))

    for dr, mult in zip(demand_ratios, multipliers):
        dr_idx = np.digitize(dr, dr_bins) - 1
        dr_idx = min(dr_idx, len(dr_bins) - 2)
        mult_idx = PRICE_MULTIPLIERS.index(mult) if mult in PRICE_MULTIPLIERS else 0
        heatmap_data[mult_idx, dr_idx] += 1

    # Normalizar por columna
    col_sums = heatmap_data.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    heatmap_norm = heatmap_data / col_sums

    im = ax.imshow(heatmap_norm, aspect="auto", cmap="Blues",
                   origin="lower", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Proporcion de uso", fontsize=11)

    ax.set_yticks(range(len(mult_labels)))
    ax.set_yticklabels(mult_labels)
    ax.set_ylabel("Multiplicador de Precio", fontsize=12)

    bin_labels = [f"{dr_bins[i]:.1f}-{dr_bins[i+1]:.1f}"
                  for i in range(len(dr_bins) - 1)]
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Ratio Demanda/Oferta", fontsize=12)

    ax.set_title("Acciones del DQN segun Demanda/Oferta",
                 fontsize=14, fontweight="bold")

    return _save(fig, "05_acciones_por_estado.png")


def plot_revenue_comparison(results: dict) -> str:
    """Barplot comparativo de revenue total."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    names = list(results.keys())
    colors_list = [COLORS.get(n, COLOR_PRIMARY) for n in names]
    x = np.arange(len(names))

    # Revenue total
    revenues = [results[n].total_revenue for n in names]
    bars = axes[0].bar(x, revenues, color=colors_list, edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.upper() for n in names], fontsize=12)
    axes[0].set_title("Revenue Total", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, revenues):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"${val:,.0f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    # Revenue promedio por viaje
    avg_revs = [results[n].avg_revenue_per_ride for n in names]
    bars = axes[1].bar(x, avg_revs, color=colors_list, edgecolor="white")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.upper() for n in names], fontsize=12)
    axes[1].set_title("Revenue Promedio por Viaje",
                      fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Revenue ($)")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, avg_revs):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"${val:.2f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")

    plt.suptitle("Comparacion de Politicas de Pricing",
                 fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    return _save(fig, "06_comparacion_revenue.png")


def plot_multiplier_distribution(results: dict) -> str:
    """Histograma de multiplicadores usados por cada politica."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (name, res) in zip(axes, results.items()):
        color = COLORS.get(name, COLOR_PRIMARY)
        ax.hist(res.multipliers_used, bins=20, color=color,
                edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(res.multipliers_used), color=COLOR_SECONDARY,
                   ls="--", lw=2,
                   label=f"Media: {np.mean(res.multipliers_used):.2f}")
        ax.set_title(f"{name.upper()}",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Multiplicador")
        ax.set_ylabel("Frecuencia")
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Distribucion de Multiplicadores de Precio",
                 fontsize=16, fontweight="bold", y=1.03)
    plt.tight_layout()
    return _save(fig, "07_distribucion_multiplicadores.png")


def build_summary_table(results: dict) -> pd.DataFrame:
    """Tabla resumen de resultados."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Politica": name.upper(),
            "Revenue Total ($)": f"{res.total_revenue:,.2f}",
            "Revenue Medio ($)": f"{res.avg_revenue_per_ride:.2f}",
            "Multiplicador Medio": f"{res.avg_multiplier:.2f}",
            "Aceptacion Media": f"{res.avg_acceptance:.2%}",
        })

    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "08_resumen_resultados.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Resumen guardado: {csv_path}")
    return summary
