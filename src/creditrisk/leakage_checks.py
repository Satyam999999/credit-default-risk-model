"""Diagnostics that support docs/LEAKAGE.md. Run: python -m creditrisk.leakage_checks --source file --path ..."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import FEATURES, TARGET
from .data import load_dataset


def assert_no_target_or_id(columns) -> None:
    banned = {TARGET, "id", "default", "default_payment_next_month"}
    bad = banned.intersection(columns)
    assert not bad, f"target/id columns present in features: {bad}"


def single_feature_auc(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Univariate AUC per feature (NaN filled with the median). DIAGNOSTIC ONLY: never used to select features.
    A value near 0 or 1 (say >0.85 or <0.15) would signal a feature that leaks the label."""
    out = {}
    for c in X.columns:
        v = X[c].astype(float)
        out[c] = roc_auc_score(y, v.fillna(v.median()))
    return pd.Series(out).sort_values(key=lambda s: (s - 0.5).abs(), ascending=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["file", "db", "synthetic"], default="file")
    ap.add_argument("--path"); ap.add_argument("--db-url")
    a = ap.parse_args(argv)
    ds = load_dataset(a.source, a.path, a.db_url)
    assert_no_target_or_id(list(ds.X.columns))   # id is only the row index, never a column
    assert list(ds.X.columns) == FEATURES
    auc = single_feature_auc(ds.X, ds.y)
    print("Top single-feature AUCs (leakage red flag if > 0.85 or < 0.15):")
    print(auc.head(10).round(3).to_string())
    flagged = auc[(auc > 0.85) | (auc < 0.15)]
    print("FLAGGED:", list(flagged.index) if len(flagged) else "none")


if __name__ == "__main__":
    main()
