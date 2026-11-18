"""Compare imbalance levers: none vs class weighting vs resampling (inside folds) vs threshold tuning.

Hyperparameters are fixed (pipelines.DEFAULT_PARAMS) so differences come from the strategy, not tuning.
For each fold we report threshold-free metrics (ROC-AUC, PR-AUC) and precision/recall at
  (a) the default 0.5 threshold and (b) a threshold tuned for the assumed cost (from training data only).
Threshold tuning is therefore a lever you can apply on top of ANY strategy; the 'none' strategy at the
tuned threshold isolates its effect.

    python -m creditrisk.imbalance_study --source file --path data/<uci file>
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import config as C
from .data import load_dataset
from .evaluation import best_threshold_cost, metrics_at_threshold, score_metrics
from .pipelines import DEFAULT_PARAMS, STRATEGIES, make_pipeline


def run(X, y, models, strategies, n_folds, n_inner, c_fn, c_fp, seed):
    rows = []
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for m in models:
        for s in strategies:
            for k, (tr, te) in enumerate(skf.split(X, y)):
                pipe = make_pipeline(m, s, seed).set_params(**DEFAULT_PARAMS[m])
                pipe.fit(X.iloc[tr], y.iloc[tr])
                p = pipe.predict_proba(X.iloc[te])[:, 1]
                inner = StratifiedKFold(n_inner, shuffle=True, random_state=seed + k)
                p_tr = cross_val_predict(clone(pipe), X.iloc[tr], y.iloc[tr], cv=inner, method="predict_proba")[:, 1]
                thr = best_threshold_cost(y.iloc[tr], p_tr, c_fn, c_fp)
                at05 = metrics_at_threshold(y.iloc[te], p, 0.5, c_fn, c_fp)
                tuned = metrics_at_threshold(y.iloc[te], p, thr, c_fn, c_fp)
                sm = score_metrics(y.iloc[te], p)
                rows.append({"model": m, "strategy": s, "fold": k, "roc_auc": sm["roc_auc"], "pr_auc": sm["pr_auc"],
                             "brier": sm["brier"],
                             **{f"{a}_at_0.5": at05[a] for a in ("precision", "recall", "f1", "cost_per_customer")},
                             "tuned_threshold": thr,
                             **{f"{a}_at_tuned": tuned[a] for a in ("precision", "recall", "f1", "cost_per_customer")}})
                print(f"[{m}] {s} fold {k+1}/{n_folds}  AUC={sm['roc_auc']:.4f}  thr={thr:.3f}", flush=True)
        print(f"[{m}] done")
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["file", "db", "synthetic"], default="file")
    ap.add_argument("--path"); ap.add_argument("--db-url")
    ap.add_argument("--models", default="lr,rf,xgb,lgbm")
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--inner", type=int, default=3)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--c-fn", type=float, default=C.COST_FN); ap.add_argument("--c-fp", type=float, default=C.COST_FP)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args(argv)
    if a.fast:
        a.folds, a.inner = 3, 2
    ds = load_dataset(a.source, a.path, a.db_url)
    df = run(ds.X, ds.y, a.models.split(","), a.strategies.split(","), a.folds, a.inner, a.c_fn, a.c_fp, a.seed)
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    df.to_csv(C.RESULTS_DIR / "imbalance_folds.csv", index=False)
    agg = df.drop(columns="fold").groupby(["model", "strategy"], sort=False).mean().round(4).reset_index()
    agg.to_csv(C.RESULTS_DIR / "imbalance_comparison.csv", index=False)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
