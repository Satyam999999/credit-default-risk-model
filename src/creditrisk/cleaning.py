"""Deterministic, row-level data cleaning plus a data-quality report.

Everything here is a FIXED rule (no statistics learned from the data), so it is
safe to run before the train/validation split. Anything that learns from the
data (scaling, imputation, encoding, resampling) lives in the sklearn Pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BILL, PAY, PAYAMT, RAW_COLUMNS, TARGET


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case names, fix the UCI quirks (PAY_0 -> pay_1, target name), check schema."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_").replace(".", "_") for c in out.columns]
    out = out.rename(columns={"pay_0": "pay_1", "default_payment_next_month": TARGET, "default": TARGET})
    missing = [c for c in RAW_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return out[RAW_COLUMNS].reset_index(drop=True)


def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map undocumented category codes to the documented 'other' bucket.

    education: documented 1-4 (4 = other); codes 0, 5, 6 appear -> 4.
    marriage : documented 1-3 (3 = other); code 0 appears        -> 3.
    """
    out = df.copy()
    out["education"] = np.where(out["education"].isin([1, 2, 3, 4]), out["education"], 4)
    out["marriage"] = np.where(out["marriage"].isin([1, 2, 3]), out["marriage"], 3)
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows identical in every column except id (keep the lowest id).

    Duplicates matter for leakage: a record copied into both the training and
    validation fold makes validation scores optimistic.
    """
    cols = [c for c in df.columns if c != "id"]
    return df.sort_values("id").drop_duplicates(subset=cols, keep="first").reset_index(drop=True)


def _any_over_months(cond_by_k: list[pd.Series]) -> pd.Series:
    return pd.concat(cond_by_k, axis=1).any(axis=1)


def data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count data-quality issues on the RAW (standardized) frame.

    Pairing convention used throughout: the payment in month k is paid toward the
    statement of the previous (older) month, i.e. pay_amt{k} vs bill_amt{k+1}.
    """
    n = len(df)
    rows: list[tuple[str, int, str]] = []

    def add(name: str, mask: pd.Series, treatment: str) -> None:
        rows.append((name, int(mask.sum()), treatment))

    add("education code outside documented 1-4", ~df["education"].isin([1, 2, 3, 4]),
        "Recoded to 4 (other); fixed rule.")
    add("marriage code outside documented 1-3", ~df["marriage"].isin([1, 2, 3]),
        "Recoded to 3 (other); fixed rule.")
    add("rows with any pay status of -2 (undocumented)", (df[PAY] == -2).any(axis=1),
        "Kept; treated as 'no delay' (max(status,0)). Commonly read as 'no consumption'.")
    add("rows with any pay status of 0 (undocumented)", (df[PAY] == 0).any(axis=1),
        "Kept; treated as 'no delay' (revolving credit / minimum paid).")
    add("rows with any pay status >= 8 (very long delay)", (df[PAY] >= 8).any(axis=1),
        "Kept; plausible but rare, trees are robust, linear model sees delay as numeric.")
    add("exact duplicate rows (ignoring id, excluding first copy)",
        df.drop(columns="id").duplicated(keep="first"),
        "Dropped before any split to avoid train/validation leakage.")
    add("rows with a negative bill amount", (df[BILL] < 0).any(axis=1),
        "Kept (credit balance); utilization uses max(bill, 0).")
    add("rows with a bill above 1.5x the credit limit",
        _any_over_months([df[b] > 1.5 * df["limit_bal"] for b in BILL]),
        "Kept (over-limit is possible); linear model uses signed-log amounts.")
    add("rows with a payment above the credit limit",
        _any_over_months([df[p] > df["limit_bal"] for p in PAYAMT]),
        "Kept; payment/bill ratio is capped at 1.")
    add("rows with a delay status but no positive previous bill",
        _any_over_months([(df[f"pay_{k}"] >= 1) & (df[f"bill_amt{k+1}"] <= 0) for k in range(1, 6)]),
        "Kept (inconsistent: nothing owed yet flagged late); ratio is undefined -> missing.")
    add("rows with status 0 but zero payment against a positive previous bill",
        _any_over_months([(df[f"pay_{k}"] == 0) & (df[f"pay_amt{k}"] == 0) & (df[f"bill_amt{k+1}"] > 0)
                          for k in range(1, 6)]),
        "Kept (status 0 should mean minimum paid); the ratio feature exposes it to the model.")
    add("rows where delay jumps from <=0 to >=2 in one month",
        _any_over_months([(df[f"pay_{k}"] >= 2) & (df[f"pay_{k+1}"] <= 0) for k in range(1, 6)]),
        "Kept (logically a delay grows by at most 1 per month, but this pattern is common in the file); reported, not removed.")
    add("rows with age outside 21-79", ~df["age"].between(21, 79), "Kept.")

    rep = pd.DataFrame(rows, columns=["check", "n_rows", "treatment"])
    rep["pct_of_rows"] = (100 * rep["n_rows"] / max(n, 1)).round(3)
    info = pd.DataFrame(
        [("total rows", n, "", 100.0),
         ("default rate (target = 1)", int(df[TARGET].sum()), "Class imbalance, see imbalance study.",
          round(100 * df[TARGET].mean(), 3))],
        columns=["check", "n_rows", "treatment", "pct_of_rows"])
    return pd.concat([info, rep], ignore_index=True)[["check", "n_rows", "pct_of_rows", "treatment"]]
