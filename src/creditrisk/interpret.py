"""Interpretability: LR coefficients, tree importances, SHAP summary/dependence, calibration.

Final models are refit (with tuning) on ALL data for explanation only. Performance and calibration
numbers always come from out-of-fold predictions produced by run_experiments, never from these refits.

    python -m creditrisk.interpret --source file --path data/<uci file> --models lr,rf,xgb,lgbm --shap-model lgbm
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from . import config as C
from .data import load_dataset
from .pipelines import SEARCH_SPACES, make_pipeline


def fit_final(X, y, model, n_iter, seed, n_jobs=-1):
    s = RandomizedSearchCV(make_pipeline(model, "none", seed), SEARCH_SPACES[model], n_iter=n_iter, scoring="roc_auc",
                           cv=StratifiedKFold(5, shuffle=True, random_state=seed), n_jobs=n_jobs,
                           random_state=seed, refit=True)
    s.fit(X, y)
    return s.best_estimator_


def lr_coefficients(pipe) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_[0]
    df = pd.DataFrame({"feature": names, "coef_per_std": coef, "odds_ratio_per_std": np.exp(coef)})
    return df.reindex(df.coef_per_std.abs().sort_values(ascending=False).index).reset_index(drop=True)


def tree_importances(pipe) -> pd.DataFrame:
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["clf"].feature_importances_
    df = pd.DataFrame({"feature": names, "importance": imp / imp.sum()})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def _positive_class(sv):
    """SHAP returns a list, a 2-D or a 3-D array depending on version/model; return (n, features) for class 1."""
    if isinstance(sv, list):
        return np.asarray(sv[-1])
    sv = np.asarray(sv)
    return sv[:, :, -1] if sv.ndim == 3 else sv


def shap_analysis(pipe, X, model, n_sample=3000, seed=C.SEED, top_dep=4):
    """Compute SHAP values using TreeExplainer.
    
    NOTE: SHAP values are computed from the final model refit on ALL training data,
    explaining a fixed sample of n_sample=3000 rows (seed=C.SEED). They are NOT
    computed out-of-fold. Performance and calibration metrics come from out-of-fold CV.
    """
    import shap
    Xs = X.sample(min(n_sample, len(X)), random_state=seed)
    Xt = pipe.named_steps["prep"].transform(Xs)
    sv = _positive_class(shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt))
    mean_abs = np.abs(sv).mean(0)
    direction = [spearmanr(Xt.iloc[:, j], sv[:, j])[0] if Xt.iloc[:, j].nunique() > 1 else np.nan
                 for j in range(Xt.shape[1])]
    tbl = pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": mean_abs, "direction_corr": direction}) \
        .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(); shap.summary_plot(sv, Xt, show=False, max_display=15)
    plt.tight_layout(); plt.savefig(C.FIGURES_DIR / f"shap_summary_{model}.png", dpi=150); plt.close()
    for f in tbl.feature.head(top_dep):
        plt.figure(); shap.dependence_plot(f, sv, Xt, show=False)
        plt.tight_layout(); plt.savefig(C.FIGURES_DIR / f"shap_dependence_{model}_{f}.png", dpi=150); plt.close()
    return tbl


def business_sense_check(tbl: pd.DataFrame) -> dict:
    """Recent delinquency should dominate, and more delay should mean higher risk."""
    top3 = list(tbl.feature.head(3))
    dominated = any(f in C.DELINQUENCY_FEATURES for f in top3)
    pay1 = tbl.loc[tbl.feature == "pay_1", "direction_corr"]
    sign_ok = bool(pay1.iloc[0] > 0) if len(pay1) and not pd.isna(pay1.iloc[0]) else None
    return {"top3": top3, "delinquency_in_top3": dominated, "pay_1_higher_delay_higher_risk": sign_ok}


def bar(df, x, y, title, path, n=15):
    d = df.head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d[y], d[x]); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["file", "db", "synthetic"], default="file")
    ap.add_argument("--path"); ap.add_argument("--db-url")
    ap.add_argument("--models", default="lr,rf,xgb,lgbm")
    ap.add_argument("--shap-model", default="lgbm")
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args(argv)
    if a.fast:
        a.n_iter = 3
    ds = load_dataset(a.source, a.path, a.db_url)
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True); C.FIGURES_DIR.mkdir(exist_ok=True, parents=True)
    for m in a.models.split(","):
        pipe = fit_final(ds.X, ds.y, m, a.n_iter, a.seed)
        if m in ("lr", "lr_binned"):
            t = lr_coefficients(pipe); t.to_csv(C.RESULTS_DIR / f"{m}_coefficients.csv", index=False)
            bar(t, "coef_per_std", "feature", f"{C.MODEL_LABELS[m]} coefficients (per 1 SD)", C.FIGURES_DIR / f"{m}_coefficients.png")
        else:
            t = tree_importances(pipe); t.to_csv(C.RESULTS_DIR / f"importances_{m}.csv", index=False)
            bar(t, "importance", "feature", f"{C.MODEL_LABELS[m]} feature importance", C.FIGURES_DIR / f"importance_{m}.png")
        if m == a.shap_model:
            tbl = shap_analysis(pipe, ds.X, m)
            tbl.to_csv(C.RESULTS_DIR / "shap_summary.csv", index=False)
            chk = business_sense_check(tbl)
            pd.Series(chk).to_json(C.RESULTS_DIR / "business_sense_check.json")
            print("Business-sense check:", chk)
    print("Calibration plot: results/figures/calibration.png (from out-of-fold predictions, made by run_experiments).")


if __name__ == "__main__":
    main()
