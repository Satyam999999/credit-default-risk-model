"""Nested stratified CV with hyperparameter tuning for each model.

Outer loop (StratifiedKFold) estimates performance. For each outer fold:
  1. RandomizedSearchCV (inner StratifiedKFold) tunes hyperparameters on the outer-TRAIN part only;
  2. the decision threshold is chosen from inner out-of-fold predictions on the outer-TRAIN part,
     minimizing the assumed cost (cost of missed default > cost of false alarm);
  3. the outer-TEST part is scored once, with that threshold.

    python -m creditrisk.run_experiments --source file --path data/<uci file> --models lr,rf,xgb,lgbm
    python -m creditrisk.run_experiments --source synthetic --models lr,rf --fast     # smoke test only
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from . import config as C
from .data import load_dataset, read_raw_file, write_dq_report
from .evaluation import best_threshold_cost, fold_ci, metrics_at_threshold, score_metrics
from .pipelines import SEARCH_SPACES, make_pipeline


def nested_cv(X, y, model, strategy, n_outer, n_inner, n_iter, scoring, c_fn, c_fp, seed, n_jobs):
    outer = StratifiedKFold(n_outer, shuffle=True, random_state=seed)
    p_oof, thr_oof, fold_id = (np.full(len(y), np.nan) for _ in range(3))
    rows, params = [], []
    for k, (tr, te) in enumerate(outer.split(X, y)):
        t0 = time.time()
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
        inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + k)
        search = RandomizedSearchCV(make_pipeline(model, strategy, seed), SEARCH_SPACES[model], n_iter=n_iter,
                                    scoring=scoring, cv=inner, n_jobs=n_jobs, random_state=seed + k, refit=True)
        search.fit(Xtr, ytr)
        best = search.best_estimator_
        p_tr = cross_val_predict(clone(best), Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
        thr = best_threshold_cost(ytr, p_tr, c_fn, c_fp)           # train-only information
        p_te = best.predict_proba(Xte)[:, 1]
        rows.append({"model": model, "fold": k, "inner_cv_score": float(search.best_score_),
                     **score_metrics(yte, p_te), **metrics_at_threshold(yte, p_te, thr, c_fn, c_fp)})
        params.append({"model": model, "fold": k,
                       "params": {a: (b.item() if hasattr(b, "item") else b) for a, b in search.best_params_.items()}})
        p_oof[te], thr_oof[te], fold_id[te] = p_te, thr, k
        print(f"[{model}] fold {k+1}/{n_outer}  AUC={rows[-1]['roc_auc']:.4f}  PR-AUC={rows[-1]['pr_auc']:.4f}  "
              f"recall={rows[-1]['recall']:.3f}  thr={thr:.3f}  ({time.time()-t0:.0f}s)")
    return rows, params, p_oof, thr_oof, fold_id


def summarize(folds: pd.DataFrame, baseline: str = "lr") -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["roc_auc", "pr_auc", "ks", "precision", "recall", "f1", "cost_per_customer", "brier"]
    srows = []
    for m, g in folds.groupby("model", sort=False):
        for met in metrics:
            mean, lo, hi = fold_ci(g[met])
            srows.append({"model": m, "metric": met, "mean": mean, "ci_low": lo, "ci_high": hi})
    summ = pd.DataFrame(srows)

    prow = []
    all_models = list(dict.fromkeys(folds["model"]))
    pairs = []
    if "lr" in all_models:
        for m in all_models:
            if m != "lr":
                pairs.append((m, "lr"))
    if "lr_binned" in all_models:
        for m in ("rf", "xgb", "lgbm"):
            if m in all_models and (m, "lr_binned") not in pairs:
                pairs.append((m, "lr_binned"))
    for m_a, m_b in [("xgb", "rf"), ("lgbm", "rf"), ("lgbm", "xgb")]:
        if m_a in all_models and m_b in all_models and (m_a, m_b) not in pairs:
            pairs.append((m_a, m_b))

    for m, b in pairs:
        gm = folds[folds.model == m].sort_values("fold")
        gb = folds[folds.model == b].sort_values("fold")
        for met in metrics:
            mean, lo, hi = fold_ci(gm[met].to_numpy() - gb[met].to_numpy())
            prow.append({
                "model": m,
                "baseline": b,
                "metric": met,
                "mean_diff_vs_baseline": mean,
                "ci_low": lo,
                "ci_high": hi,
            })
    return summ, pd.DataFrame(prow)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["file", "db", "synthetic"], default="file")
    ap.add_argument("--path"); ap.add_argument("--db-url")
    ap.add_argument("--models", default="lr,lr_binned,rf,xgb,lgbm")
    ap.add_argument("--strategy", default="none", choices=["none", "class_weight", "undersample", "smote"])
    ap.add_argument("--outer", type=int, default=5); ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--scoring", default="roc_auc")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--c-fn", type=float, default=C.COST_FN); ap.add_argument("--c-fp", type=float, default=C.COST_FP)
    ap.add_argument("--fast", action="store_true", help="3 outer folds, 2 inner folds, 3 search iterations")
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", default=str(C.RESULTS_DIR))
    a = ap.parse_args(argv)
    if a.fast:
        a.outer, a.inner, a.n_iter = 3, 2, 3

    ds = load_dataset(a.source, a.path, a.db_url)
    if a.source == "file":
        write_dq_report(read_raw_file(a.path), C.DOCS_DIR / "DATA_QUALITY_FINDINGS.md")
    print(f"rows: raw={ds.n_raw} clean={ds.n_clean}  default rate={ds.y.mean():.3%}  features={ds.X.shape[1]}")
    out = C.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    models = a.models.split(",")
    all_rows, all_params = [], []
    oof = pd.DataFrame({"y_true": ds.y.to_numpy()}, index=ds.X.index)
    for m in models:
        rows, params, p, thr, fid = nested_cv(ds.X, ds.y, m, a.strategy, a.outer, a.inner, a.n_iter, a.scoring,
                                              a.c_fn, a.c_fp, a.seed, a.n_jobs)
        all_rows += rows; all_params += params
        oof[f"p_{m}"], oof[f"thr_{m}"] = p, thr
        oof["fold"] = fid
    folds = pd.DataFrame(all_rows)
    summ, paired = summarize(folds, baseline="lr")
    folds.to_csv(out / "cv_fold_metrics.csv", index=False)
    summ.to_csv(out / "cv_summary.csv", index=False)
    paired.to_csv(out / "cv_paired_vs_baseline.csv", index=False)
    oof.to_csv(out / "oof_predictions.csv")
    (out / "best_params.json").write_text(json.dumps(all_params, indent=2, default=str))
    (out / "run_config.json").write_text(json.dumps(
        {**vars(a), "n_raw": ds.n_raw, "n_clean": ds.n_clean, "source": ds.source,
         "cost_assumption": "ASSUMPTION: missed default costs c_fn, false alarm costs c_fp"}, indent=2))
    print(summ.pivot(index="model", columns="metric", values="mean").round(4).to_string())

    from . import plots
    ms = [m for m in models]
    plots.plot_roc_pr(oof, ms, C.FIGURES_DIR / "roc_pr.png")
    plots.plot_gain_lift(oof, ms, C.FIGURES_DIR / "gain_lift.png")
    plots.plot_confusion(oof, ms, C.FIGURES_DIR / "confusion_matrices.png")
    plots.plot_calibration(oof, ms, C.FIGURES_DIR / "calibration.png")
    plots.plot_ks(oof, ms[-1], C.FIGURES_DIR / f"ks_{ms[-1]}.png")


if __name__ == "__main__":
    main()
