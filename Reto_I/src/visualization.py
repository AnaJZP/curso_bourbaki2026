"""
visualization.py -- Generacion de todas las graficas del analisis.
Paleta unificada: escala de azules.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from .config import CMAP_HEATMAP, COLOR_CLASS_0, COLOR_CLASS_1, COLORS, OUTPUT_DIR, SEED


# -- Utilidad interna -----------------------------------------

def _save(fig, filename: str) -> str:
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Guardado: {path}")
    return str(path)


# -- Graficas -------------------------------------------------

def plot_samples(df: pd.DataFrame) -> str:
    """Muestra 5 memes benignos y 5 ofensivos."""
    fig, axes = plt.subplots(2, 5, figsize=(22, 10))
    fig.suptitle(
        "Muestras del Dataset -- Hateful Memes Challenge (Meta AI)",
        fontsize=16, fontweight="bold",
    )

    for row_axes, label, color, title in [
        (axes[0], 0, COLOR_CLASS_0, "NO OFENSIVO"),
        (axes[1], 1, COLOR_CLASS_1, "OFENSIVO"),
    ]:
        sample = df[df["label"] == label].sample(5, random_state=SEED)
        for ax, (_, row) in zip(row_axes, sample.iterrows()):
            img = Image.open(row["img_path"])
            ax.imshow(img)
            ax.set_title(title, fontsize=11, color=color, fontweight="bold")
            txt = str(row["text"])[:50] + ("..." if len(str(row["text"])) > 50 else "")
            ax.set_xlabel(txt, fontsize=8, wrap=True)
            ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    return _save(fig, "01_muestras_dataset.png")


def plot_class_distribution(df: pd.DataFrame) -> str:
    """Barplot de distribucion de clases."""
    counts = df["label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["clase"] = counts["label"].map({0: "No ofensivo", 1: "Ofensivo"})

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        counts["clase"], counts["count"],
        color=[COLOR_CLASS_0, COLOR_CLASS_1], edgecolor="white", linewidth=1.5,
    )
    for i, cnt in enumerate(counts["count"]):
        ax.text(i, cnt + 2, str(cnt), ha="center", fontsize=14, fontweight="bold")

    ax.set_title("Distribucion de Clases", fontsize=16, fontweight="bold")
    ax.set_ylabel("Cantidad")
    ax.grid(axis="y", alpha=0.3)

    return _save(fig, "02_distribucion_clases.png")


def plot_text_length(df: pd.DataFrame) -> str:
    """Box plot de longitud de texto por clase."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x="clase", y="text_length",
        palette={"No ofensivo": COLOR_CLASS_0, "Ofensivo": COLOR_CLASS_1}, ax=ax,
    )
    ax.set_title("Distribucion de Longitud de Texto por Clase",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Longitud del texto (caracteres)")
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, "03_longitud_texto.png")


def plot_learning_curves(results: dict) -> str:
    """Curvas de perdida train/val para cada estrategia."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, res) in zip(axes, results.items()):
        h = res.history
        ax.plot(h["train_loss"], label="Train", color=COLORS[name], lw=2)
        ax.plot(h["val_loss"], label="Val", color=COLORS[name], lw=2, ls="--")
        ax.set_title(f"Fusion: {name.upper()}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoca"); ax.set_ylabel("Perdida (BCE)")
        ax.legend(); ax.grid(alpha=0.3)
    plt.suptitle("Curvas de Aprendizaje", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "04_curvas_aprendizaje.png")


def plot_roc_curves(results: dict) -> str:
    """Curvas ROC comparativas."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, res), c in zip(results.items(), COLORS.values()):
        fpr, tpr, _ = roc_curve(res.labels, res.preds)
        auc = roc_auc_score(res.labels, res.preds)
        ax.plot(fpr, tpr, label=f"{name.upper()} (AUROC={auc:.3f})", color=c, lw=3)
    ax.plot([0, 1], [0, 1], color="gray", lw=1, ls="--", label="Random")
    ax.set_title("Curvas ROC -- Comparacion de Estrategias",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("FPR", fontsize=14); ax.set_ylabel("TPR", fontsize=14)
    ax.legend(loc="lower right", fontsize=11); ax.grid(alpha=0.3)
    return _save(fig, "05_curvas_roc.png")


def plot_confusion_matrices(results: dict) -> str:
    """Matrices de confusion lado a lado."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (name, res) in zip(axes, results.items()):
        bp = (res.preds > 0.5).astype(int)
        cm = confusion_matrix(res.labels, bp)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=CMAP_HEATMAP, ax=ax,
            xticklabels=["Benigno", "Ofensivo"],
            yticklabels=["Benigno", "Ofensivo"],
        )
        acc = accuracy_score(res.labels, bp)
        ax.set_title(f"{name.upper()} -- Acc: {acc:.2%}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Prediccion"); ax.set_ylabel("Real")
    plt.suptitle("Matrices de Confusion", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _save(fig, "06_matrices_confusion.png")


def build_summary_table(
    results: dict,
    fusions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Genera y guarda tabla resumen de resultados."""
    rows = []
    for name, res in results.items():
        bp = (res.preds > 0.5).astype(int)
        rows.append({
            "Estrategia":   name.upper(),
            "AUROC":        f"{roc_auc_score(res.labels, res.preds):.4f}",
            "Accuracy":     f"{accuracy_score(res.labels, bp):.4f}",
            "Dim. Input":   fusions[name].shape[1],
            "Best Val Loss": f"{min(res.history['val_loss']):.4f}",
        })
    summary = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "07_resumen_resultados.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  Resumen guardado: {csv_path}")
    return summary
