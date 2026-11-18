"""Metrics, KS, gain/lift, cost-based threshold, and fold confidence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (average_precision_score, brier_score_loss, roc_auc_score, roc_curve)


def ks_statistic(y, p) -> float:
    """Max gap between the cumulative score distributions of defaulters and non-defaulters."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def confusion_counts(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {"tn": int(((y == 0) & (pred == 0)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "fn": int(((y == 1) & (pred == 0)).sum()), "tp": int(((y == 1) & (pred == 1)).sum())}


def metrics_at_threshold(y, p, thr: float, c_fn: float, c_fp: float) -> dict:
    p = np.asarray(p)
    c = confusion_counts(y, p >= thr)
    prec = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    rec = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    cost = (c_fn * c["fn"] + c_fp * c["fp"]) / len(p)
    return {"threshold": float(thr), "precision": prec, "recall": rec, "f1": f1,
            "cost_per_customer": cost, **c}


def score_metrics(y, p) -> dict:
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "ks": ks_statistic(y, p), "brier": float(brier_score_loss(y, p))}


def best_threshold_cost(y, p, c_fn: float, c_fp: float) -> float:
    """Threshold t (predict default if p >= t) minimizing c_fn*FN + c_fp*FP on the given data."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    last_of_tie = np.r_[ps[1:] != ps[:-1], True]          # only cut between distinct scores
    fn = ys.sum() - tp
    cost = c_fn * fn + c_fp * fp
    cost_none = c_fn * ys.sum()                            # predict nobody
    idx = np.flatnonzero(last_of_tie)
    best = idx[np.argmin(cost[idx])]
    if cost[best] >= cost_none:
        return float(np.nextafter(ps[0], np.inf))          # flag nobody
    return float(ps[best])


def gain_lift_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: customers ranked by score (highest risk first)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    df["bin"] = np.floor(np.arange(len(df)) * n_bins / len(df)).astype(int) + 1
    g = df.groupby("bin")["y"].agg(n="size", defaults="sum").reset_index()
    base = df["y"].mean()
    g["default_rate"] = g["defaults"] / g["n"]
    g["lift"] = g["default_rate"] / base
    g["cum_defaults_captured"] = g["defaults"].cumsum() / df["y"].sum()      # gain
    g["cum_pct_customers"] = g["n"].cumsum() / len(df)
    g["cum_lift"] = g["cum_defaults_captured"] / g["cum_pct_customers"]
    return g


def fold_ci(values, level: float = 0.95) -> tuple[float, float, float]:
    """Mean and t-interval across folds. Folds share training data, so this interval
    is somewhat optimistic (Nadeau & Bengio 2003); treat it as indicative."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    m = float(v.mean())
    if n < 2:
        return m, np.nan, np.nan
    half = stats.t.ppf(0.5 + level / 2, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return m, m - half, m + half


def cost_sensitivity(y, p, ratios=(2, 5, 10)) -> pd.DataFrame:
    """Illustrative only: thresholds are optimized on the same predictions they are scored on."""
    rows = []
    for r in ratios:
        thr = best_threshold_cost(y, p, r, 1.0)
        rows.append({"cost_ratio_fn_to_fp": r, **metrics_at_threshold(y, p, thr, r, 1.0)})
    return pd.DataFrame(rows)
