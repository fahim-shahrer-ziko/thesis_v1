"""
evaluate.py  —  Stage 7
=========================
Compute all evaluation metrics on the test-set predictions and save:
  - evaluation_report.txt       — text summary of all metrics
  - confusion_matrix.png        — counts + row-normalised % heatmaps
  - mode_distribution.png       — true vs predicted mode bar chart
  - persona_accuracy.png        — accuracy bar chart by persona
  - persona_accuracy_breakdown.csv — per-persona metric table

Input : outputs/test_predictions_full.csv
Output: all files written to outputs/
"""

import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server/CI environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import OUTPUTS_DIR, TRAVELER_ID_COL
from utils import get_logger

logger = get_logger(__name__, log_file="evaluate.log")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_predictions() -> pd.DataFrame:
    """Load the predictions CSV produced by test_predict.py."""
    path = f"{OUTPUTS_DIR}test_predictions_full.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run test_predict.py first."
        )
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} predictions from {path}")

    # Validate required columns
    for col in ["true_mode", "predicted_mode"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# OVERALL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_overall_metrics(df: pd.DataFrame) -> dict:
    """Compute accuracy, macro F1, weighted F1, and per-class report."""
    y_true = df["true_mode"]
    y_pred = df["predicted_mode"]

    acc     = accuracy_score(y_true, y_pred)
    f1_mac  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wgt  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report  = classification_report(y_true, y_pred, zero_division=0)

    logger.info("=" * 60)
    logger.info("OVERALL METRICS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy     : {acc:.4f}  ({acc*100:.2f}%)")
    logger.info(f"  F1 (macro)   : {f1_mac:.4f}")
    logger.info(f"  F1 (weighted): {f1_wgt:.4f}")
    logger.info(f"\nPer-class report:\n{report}")

    return {"accuracy": acc, "f1_macro": f1_mac, "f1_weighted": f1_wgt, "report": report}


# ─────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(df: pd.DataFrame, save_path: str):
    """
    Save a 2-panel confusion matrix figure:
      Left  — raw counts
      Right — row-normalised percentages (recall per class)
    """
    labels = sorted(df["true_mode"].unique())
    cm     = confusion_matrix(df["true_mode"], df["predicted_mode"], labels=labels)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes, [cm, cm_pct], ["d", ".1f"], ["Counts", "Row %"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax,
        )
        ax.set_title(f"Confusion Matrix ({title})", fontsize=13)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True",      fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved confusion matrix → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MODE DISTRIBUTION COMPARISON PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_mode_distribution(df: pd.DataFrame, save_path: str):
    """Bar chart comparing true vs. predicted mode frequencies."""
    modes = sorted(df["true_mode"].unique())
    x     = np.arange(len(modes))
    w     = 0.35

    true_counts = [df["true_mode"].eq(m).sum()      for m in modes]
    pred_counts = [df["predicted_mode"].eq(m).sum() for m in modes]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, true_counts, w, label="True",      color="steelblue")
    ax.bar(x + w / 2, pred_counts, w, label="Predicted", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylabel("Trip Count")
    ax.set_title("True vs Predicted Mode Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved mode distribution → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# PERSONA-LEVEL METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_persona_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute accuracy and macro F1 broken down by assigned persona label."""
    if "persona_label" not in df.columns:
        logger.warning("No persona_label column found; skipping persona breakdown.")
        return pd.DataFrame()

    records = []
    for persona, g in df.groupby("persona_label"):
        acc_p = accuracy_score(g["true_mode"], g["predicted_mode"])
        f1_p  = f1_score(g["true_mode"], g["predicted_mode"], average="macro", zero_division=0)
        records.append({
            "persona":       persona,
            "n_trips":       len(g),
            "accuracy":      round(acc_p, 4),
            "f1_macro":      round(f1_p, 4),
            "top_true_mode": g["true_mode"].mode()[0],
            "top_pred_mode": g["predicted_mode"].mode()[0],
        })

    pdf = pd.DataFrame(records).sort_values("accuracy", ascending=False)
    logger.info("PERSONA-LEVEL ACCURACY:")
    logger.info(pdf.to_string(index=False))
    return pdf


# ─────────────────────────────────────────────────────────────────────────────
# PERSONA ACCURACY BAR CHART
# ─────────────────────────────────────────────────────────────────────────────

def plot_persona_accuracy(pdf: pd.DataFrame, save_path: str):
    """Horizontal bar chart of accuracy per persona, with mean reference line."""
    if pdf.empty:
        return

    mean_acc = pdf["accuracy"].mean()
    colors   = ["steelblue" if a >= mean_acc else "salmon" for a in pdf["accuracy"]]

    fig, ax = plt.subplots(figsize=(10, max(5, len(pdf) * 0.7)))
    bars    = ax.barh(pdf["persona"], pdf["accuracy"] * 100, color=colors)
    ax.axvline(mean_acc * 100, color="black", linestyle="--",
               label=f"Mean {mean_acc*100:.1f}%")
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Prediction Accuracy by Traveler Persona")
    ax.legend()

    # Annotate each bar with trip count
    for bar, n in zip(bars, pdf["n_trips"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"n={n}", va="center", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved persona accuracy chart → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TEXT REPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_text_report(metrics: dict, persona_df: pd.DataFrame,
                     df: pd.DataFrame, save_path: str):
    """Write a human-readable evaluation report to a text file."""
    lines = [
        "=" * 60,
        "APOLLO MODE CHOICE — EVALUATION REPORT",
        "=" * 60,
        f"\nTest trips      : {len(df)}",
        f"Test travelers  : {df[TRAVELER_ID_COL].nunique() if TRAVELER_ID_COL in df.columns else 'N/A'}",
        "",
        f"Accuracy        : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)",
        f"F1 (macro)      : {metrics['f1_macro']:.4f}",
        f"F1 (weighted)   : {metrics['f1_weighted']:.4f}",
        "",
        "Per-class classification report:",
        metrics["report"],
    ]
    if not persona_df.empty:
        lines += ["", "Persona-level accuracy:", persona_df.to_string(index=False)]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved text report → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("STAGE 7: EVALUATION")
    logger.info("=" * 60)

    df = load_predictions()

    # Overall metrics
    metrics = compute_overall_metrics(df)

    # Persona breakdown
    persona_df = compute_persona_metrics(df)

    # Plots
    plot_confusion_matrix(df,    f"{OUTPUTS_DIR}confusion_matrix.png")
    plot_mode_distribution(df,   f"{OUTPUTS_DIR}mode_distribution.png")
    plot_persona_accuracy(persona_df, f"{OUTPUTS_DIR}persona_accuracy.png")

    # Save persona CSV
    if not persona_df.empty:
        persona_df.to_csv(f"{OUTPUTS_DIR}persona_accuracy_breakdown.csv", index=False)
        logger.info(f"Saved persona breakdown → {OUTPUTS_DIR}persona_accuracy_breakdown.csv")

    # Text report
    save_text_report(metrics, persona_df, df, f"{OUTPUTS_DIR}evaluation_report.txt")

    logger.info("=" * 60)
    logger.info("STAGE 7 COMPLETE — all outputs saved to outputs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
