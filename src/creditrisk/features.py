"""pandas mirror of sql/03_features.sql.

The SQL views are the primary implementation (window functions in PostgreSQL).
This mirror lets the project run without a database and is used to unit-test the
feature definitions and to cross-check the SQL output (tests/test_sql_parity.py).

All features are computed per customer from that customer's own 6 months only.
No cross-customer statistic is used, so computing them before the CV split
cannot leak information between folds.

Time axis: t = 1 (oldest, April) ... 6 (newest, September); UCI column k maps to t = 7 - k.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .config import ENGINEERED

T = np.arange(1, 7, dtype=float)


def _chron(df: pd.DataFrame, prefix: str) -> np.ndarray:
    """(n, 6) array ordered oldest -> newest."""
    return df[[f"{prefix}{k}" for k in range(6, 0, -1)]].to_numpy(dtype=float)


def masked_slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-row OLS slope of y on x, ignoring NaN pairs (== PostgreSQL REGR_SLOPE)."""
    xb = np.broadcast_to(x, y.shape).astype(float)
    mask = ~np.isnan(y)
    n = mask.sum(axis=1).astype(float)
    xm, ym = np.where(mask, xb, 0.0), np.where(mask, y, 0.0)
    sx, sy = xm.sum(1), ym.sum(1)
    sxx, sxy = (xm * xm).sum(1), (xm * ym).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        den = n * sxx - sx ** 2
        slope = (n * sxy - sx * sy) / den
    slope[(n < 2) | (den == 0)] = np.nan
    return slope


def _nanmean(a: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=1)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df plus the engineered columns listed in config.ENGINEERED."""
    pay, bill, pamt = _chron(df, "pay_"), _chron(df, "bill_amt"), _chron(df, "pay_amt")
    limit = df["limit_bal"].to_numpy(dtype=float)

    prev_bill = np.full_like(bill, np.nan)
    prev_bill[:, 1:] = bill[:, :-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(prev_bill > 0, np.clip(pamt / prev_bill, 0.0, 1.0), np.nan)
        util = np.maximum(bill, 0.0) / np.where(limit > 0, limit, np.nan)[:, None]
    delay = np.maximum(pay, 0.0)
    delinq = (pay >= 1).astype(float)

    out = df.copy()
    out["avg_pay_ratio"] = _nanmean(ratio)
    out["pay_ratio_trend"] = masked_slope(ratio, T)
    out["pay_ratio_last3"] = _nanmean(ratio[:, 3:])
    out["n_months_delinquent"] = delinq.sum(1)
    out["n_delinquent_last3"] = delinq[:, 3:].sum(1)
    out["max_delay"] = delay.max(1)
    out["delay_trend"] = masked_slope(delay, T)
    out["avg_utilization"] = _nanmean(util)
    out["utilization_latest"] = util[:, -1]
    out["utilization_trend"] = masked_slope(util, T)
    assert all(c in out.columns for c in ENGINEERED)
    return out
