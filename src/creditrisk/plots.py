"""Matplotlib figures built from out-of-fold predictions (never in-sample scores)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import MODEL_LABELS
from .evaluation import confusion_counts, gain_lift_table, ks_statistic


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_pr(oof: pd.DataFrame, models, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    y = oof["y_true"].to_numpy()
    for m in models:
        fpr, tpr, _ = roc_curve(y, oof[f"p_{m}"])
        pr, rc, _ = precision_recall_curve(y, oof[f"p_{m}"])
        ax[0].plot(fpr, tpr, label=MODEL_LABELS[m])
        ax[1].plot(rc, pr, label=MODEL_LABELS[m])
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (out-of-fold)")
    ax[1].axhline(y.mean(), color="k", ls="--", lw=.8)
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (out-of-fold)")
    ax[0].legend(fontsize=8)
    _save(fig, path)


def plot_gain_lift(oof: pd.DataFrame, models, path: Path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    y = oof["y_true"].to_numpy()
    for m in models:
        t = gain_lift_table(y, oof[f"p_{m}"])
        ax[0].plot(np.r_[0, t["cum_pct_customers"]], np.r_[0, t["cum_defaults_captured"]], marker="o", ms=3,
                   label=MODEL_LABELS[m])
        ax[1].plot(t["bin"], t["lift"], marker="o", ms=3, label=MODEL_LABELS[m])
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set(xlabel="Share of customers contacted (riskiest first)", ylabel="Share of defaults captured",
              title="Cumulative gain")
    ax[1].axhline(1, color="k", ls="--", lw=.8)
    ax[1].set(xlabel="Risk decile (1 = riskiest)", ylabel="Lift vs average", title="Lift by decile")
    ax[0].legend(fontsize=8)
    _save(fig, path)


def plot_ks(oof: pd.DataFrame, model: str, path: Path):
    y, p = oof["y_true"].to_numpy(), oof[f"p_{model}"].to_numpy()
    grid = np.linspace(0, 1, 201)
    c1 = np.array([(p[y == 1] <= g).mean() for g in grid])
    c0 = np.array([(p[y == 0] <= g).mean() for g in grid])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(grid, c0, label="Non-defaulters"); ax.plot(grid, c1, label="Defaulters")
    ax.set(xlabel="Predicted probability", ylabel="Cumulative share",
           title=f"KS = {ks_statistic(y, p):.3f} ({MODEL_LABELS[model]})")
    ax.legend()
    _save(fig, path)


def plot_confusion(oof: pd.DataFrame, models, path: Path):
    fig, axes = plt.subplots(1, len(models), figsize=(3.6 * len(models), 3.6), squeeze=False)
    y = oof["y_true"].to_numpy()
    for ax, m in zip(axes[0], models):
        c = confusion_counts(y, oof[f"p_{m}"] >= oof[f"thr_{m}"])
        mat = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
               yticklabels=["No default", "Default"], xlabel="Predicted", ylabel="Actual",
               title=MODEL_LABELS[m].replace(" (baseline)", "\n(baseline)"))
    _save(fig, path)


def plot_calibration(oof: pd.DataFrame, models, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    y = oof["y_true"].to_numpy()
    for m in models:
        frac, mean_p = calibration_curve(y, oof[f"p_{m}"], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=3, label=MODEL_LABELS[m])
    ax.plot([0, 1], [0, 1], "k--", lw=.8)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed default rate", title="Calibration (out-of-fold)")
    ax.legend(fontsize=8)
    _save(fig, path)
