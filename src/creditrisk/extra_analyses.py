"""Analyses that follow the nested CV: operational ranking, cost-ratio sensitivity, grouped SHAP.

These produce the numbers quoted in the README and the report, so every figure there can be regenerated:

    python -m creditrisk.extra_analyses all                 # everything (re-fits models for the cost sensitivity)
    python -m creditrisk.extra_analyses ranking             # needs results/oof_predictions.csv only
    python -m creditrisk.extra_analyses shap-groups         # needs results/shap_summary.csv only
    python -m creditrisk.extra_analyses cost-summary        # re-summarise results/cost_sensitivity_folds.csv
    python -m creditrisk.extra_analyses cost-refit          # recompute the per-fold costs (needs the data)

Leakage note for the cost sensitivity: for every outer fold and every FN:FP ratio the threshold is chosen
from inner out-of-fold predictions on the outer-TRAINING rows only, exactly as in run_experiments.nested_cv.
At the ratio used in the main run, the refit must reproduce cv_fold_metrics.csv; a mismatch is reported.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C
from .data import load_dataset
from .evaluation import best_threshold_cost, fold_ci, metrics_at_threshold
from .pipelines import make_pipeline

RATIOS = (2.0, 5.0, 10.0)
QUANTILES = (0.1, 0.2, 0.3)


# --------------------------------------------------------------------------- operational ranking
def n_top(n: int, q: float) -> int:
    """Number of customers in the top q share of a fold, rounded to the nearest customer."""
    return int(np.floor(q * n + 0.5))


def defaults_in_top(y, p, q: float) -> tuple[int, int]:
    """(defaults captured, customers selected) when flagging the top q share by predicted risk."""
    y, p = np.asarray(y), np.asarray(p)
    k = n_top(len(y), q)
    idx = np.argsort(-p, kind="mergesort")[:k]
    return int(y[idx].sum()), k


def operational_ranking(oof: pd.DataFrame, baseline: str, challenger: str, quantiles=QUANTILES):
    """Per-fold and summary comparison of defaults captured in the top q of risk scores."""
    rows = []
    for q in quantiles:
        for k, g in oof.groupby("fold"):
            y = g["y_true"].to_numpy()
            cb, nt = defaults_in_top(y, g[f"p_{baseline}"], q)
            cc, _ = defaults_in_top(y, g[f"p_{challenger}"], q)
            d = int(y.sum())
            rows.append({"top_share": q, "fold": int(k), "customers": len(y), "defaults": d, "n_top": nt,
                         f"captured_{baseline}": cb, f"captured_{challenger}": cc,
                         "diff_defaults": cc - cb, f"pct_{baseline}": 100 * cb / d, f"pct_{challenger}": 100 * cc / d,
                         "diff_pp": 100 * (cc - cb) / d})
    folds = pd.DataFrame(rows)
    srows = []
    for q, g in folds.groupby("top_share"):
        m_d, lo_d, hi_d = fold_ci(g["diff_defaults"])
        m_p, lo_p, hi_p = fold_ci(g["diff_pp"])
        srows.append({"top_share": q, "baseline": baseline, "challenger": challenger,
                      "defaults_per_fold": g["defaults"].mean(),
                      f"captured_{baseline}": g[f"captured_{baseline}"].mean(),
                      f"captured_{challenger}": g[f"captured_{challenger}"].mean(),
                      f"pct_{baseline}": g[f"pct_{baseline}"].mean(), f"pct_{challenger}": g[f"pct_{challenger}"].mean(),
                      "diff_defaults": m_d, "diff_defaults_ci_low": lo_d, "diff_defaults_ci_high": hi_d,
                      "diff_pp": m_p, "diff_pp_ci_low": lo_p, "diff_pp_ci_high": hi_p})
    return folds, pd.DataFrame(srows)


# --------------------------------------------------------------------------- grouped SHAP
def grouped_shap(shap_df: pd.DataFrame) -> dict:
    """Summed mean |SHAP| for the delinquency family vs everything else (transformed-feature level)."""
    base = shap_df["feature"].str.replace(r"_missingindicator.*$", "", regex=True)
    is_d = base.isin(C.DELINQUENCY_GROUP)
    d, r = float(shap_df.loc[is_d, "mean_abs_shap"].sum()), float(shap_df.loc[~is_d, "mean_abs_shap"].sum())
    top = shap_df.sort_values("mean_abs_shap", ascending=False).iloc[0]
    return {"n_features": int(len(shap_df)),
            "delinquency": {"n_features": int(is_d.sum()), "sum_mean_abs_shap": d, "share": d / (d + r)},
            "other": {"n_features": int((~is_d).sum()), "sum_mean_abs_shap": r, "share": r / (d + r)},
            "top_feature": str(top["feature"]), "top_feature_mean_abs_shap": float(top["mean_abs_shap"])}


# --------------------------------------------------------------------------- cost-ratio sensitivity
def summarize_cost_sensitivity(folds: pd.DataFrame, baseline: str = "lr_binned") -> pd.DataFrame:
    """Mean cost with a fold interval, and the paired difference against `baseline`, for each ratio."""
    rows = []
    for r, gr in folds.groupby("fn_cost"):
        base = gr[gr.model == baseline].sort_values("fold")["cost_per_customer"].to_numpy()
        for m, g in gr.groupby("model"):
            c = g.sort_values("fold")["cost_per_customer"].to_numpy()
            mean, lo, hi = fold_ci(c)
            dm, dlo, dhi = fold_ci(c - base) if len(base) == len(c) else (np.nan,) * 3
            rows.append({"fn_cost": r, "model": m, "mean_cost": mean, "ci_low": lo, "ci_high": hi,
                         f"diff_vs_{baseline}": dm, "diff_ci_low": dlo, "diff_ci_high": dhi})
    return pd.DataFrame(rows).sort_values(["fn_cost", "model"]).reset_index(drop=True)


def refit_fold_costs(X, y, oof, best_params, cfg, ratios=RATIOS, c_fp=C.COST_FP, models=None) -> pd.DataFrame:
    """Re-select the threshold per outer fold and ratio from TRAIN-only inner predictions; score the test fold."""
    models = models or [c[2:] for c in oof.columns if c.startswith("p_")]
    seed, n_inner, strategy = cfg["seed"], cfg["inner"], cfg.get("strategy", "none")
    rows = []
    for m in models:
        for k in sorted(oof["fold"].unique()):
            te = (oof["fold"] == k).to_numpy()
            params = next(p["params"] for p in best_params if p["model"] == m and p["fold"] == int(k))
            pipe = make_pipeline(m, strategy, seed).set_params(**params)
            inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + int(k))
            p_tr = cross_val_predict(clone(pipe), X[~te], y[~te], cv=inner, method="predict_proba")[:, 1]
            p_te = oof.loc[te, f"p_{m}"].to_numpy()
            for r in ratios:
                thr = best_threshold_cost(y[~te], p_tr, r, c_fp)
                mt = metrics_at_threshold(y[te], p_te, thr, r, c_fp)
                rows.append({"model": m, "fold": int(k), "fn_cost": float(r), "threshold": thr,
                             "cost_per_customer": mt["cost_per_customer"], "recall": mt["recall"],
                             "precision": mt["precision"]})
            print(f"[cost-refit] {m} fold {int(k) + 1}", flush=True)
    return pd.DataFrame(rows)


def check_against_main_run(folds: pd.DataFrame, fold_metrics: pd.DataFrame, c_fn: float, tol: float = 1e-9):
    """The refit at the main-run ratio must reproduce cv_fold_metrics.csv. Returns (max abs diff, ok)."""
    a = folds[folds.fn_cost == c_fn].set_index(["model", "fold"])["cost_per_customer"]
    b = fold_metrics.set_index(["model", "fold"])["cost_per_customer"].reindex(a.index)
    diff = float((a - b).abs().max())
    return diff, diff <= tol


# --------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "ranking", "shap-groups", "cost-summary", "cost-refit"])
    ap.add_argument("--source", choices=["file", "db", "synthetic"], default="file")
    ap.add_argument("--results", default=str(C.RESULTS_DIR), help="folder with the result files (default: results/)")
    ap.add_argument("--path"); ap.add_argument("--db-url")
    ap.add_argument("--rank-baseline", default="lr_binned"); ap.add_argument("--rank-challenger", default="lgbm")
    ap.add_argument("--models", default=None, help="models for cost-refit (default: all in oof_predictions.csv)")
    a = ap.parse_args(argv)
    R = C.Path(a.results)
    steps = ["ranking", "shap-groups", "cost-refit", "cost-summary"] if a.step == "all" else [a.step]

    if "ranking" in steps:
        oof = pd.read_csv(R / "oof_predictions.csv", index_col="id")
        folds, summ = operational_ranking(oof, a.rank_baseline, a.rank_challenger)
        folds.to_csv(R / "operational_ranking_folds.csv", index=False)
        summ.to_csv(R / "operational_ranking.csv", index=False)
        print(summ.round(3).to_string(index=False))
    if "shap-groups" in steps and not (R / "shap_summary.csv").exists() and a.step == "all":
        print("shap-groups skipped: results/shap_summary.csv not found (run `make interpret` first).")
    elif "shap-groups" in steps:
        out = grouped_shap(pd.read_csv(R / "shap_summary.csv"))
        (R / "shap_groups.json").write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
    if "cost-refit" in steps:
        cfg = json.loads((R / "run_config.json").read_text())
        oof = pd.read_csv(R / "oof_predictions.csv", index_col="id")
        ds = load_dataset(a.source, a.path, a.db_url)
        oof = oof.loc[ds.X.index]
        folds = refit_fold_costs(ds.X, ds.y.to_numpy(), oof, json.loads((R / "best_params.json").read_text()), cfg,
                                 c_fp=cfg["c_fp"], ratios=sorted({*RATIOS, cfg["c_fn"]}),
                                 models=a.models.split(",") if a.models else None)
        diff, ok = check_against_main_run(folds, pd.read_csv(R / "cv_fold_metrics.csv"), cfg["c_fn"])
        print(f"Self-check vs cv_fold_metrics.csv at ratio {cfg['c_fn']:g}: max |diff| = {diff:.2e} "
              f"({'OK' if ok else 'MISMATCH - investigate before using these numbers'})")
        folds.to_csv(R / "cost_sensitivity_folds.csv", index=False)
    if "cost-summary" in steps:
        folds = pd.read_csv(R / "cost_sensitivity_folds.csv")
        summ = summarize_cost_sensitivity(folds)
        summ.to_csv(R / "cost_sensitivity.csv", index=False)
        print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
