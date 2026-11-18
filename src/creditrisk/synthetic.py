"""Synthetic UCI-shaped data, ONLY for smoke tests and unit tests.

It is NOT the real dataset: never report model results from it. It mimics the
schema and the data-quality quirks (invalid codes, -2/0 statuses, negative
bills, duplicates) so the pipeline can be exercised without a download.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_COLUMNS, TARGET


def make_synthetic(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice([10000, 20000, 50000, 100000, 200000, 300000], n,
                       p=[.05, .2, .2, .25, .2, .1]).astype(float)
    risk = rng.normal(size=n)
    sig = lambda z: 1 / (1 + np.exp(-z))

    state = np.where(rng.random(n) < sig(risk - 1.2), rng.integers(1, 4, n),
                     rng.choice([-2, -1, 0], n, p=[.2, .25, .55]))
    status = np.zeros((n, 6), dtype=int)           # oldest -> newest
    for t in range(6):
        u = rng.random(n)
        late = state >= 1
        nxt_late = np.where(u < .5, np.minimum(state + 1, 8), np.where(u < .8, state, 0))
        nxt_ok = np.where(rng.random(n) < sig(risk - 2), 1, rng.choice([-2, -1, 0], n, p=[.2, .25, .55]))
        state = np.where(late, nxt_late, nxt_ok)
        status[:, t] = state

    bill = limit[:, None] * np.clip(rng.beta(2, 3, (n, 6)) + 0.1 * risk[:, None], 0, 1.3)
    frac = np.where(status >= 1, rng.uniform(0, .1, (n, 6)), rng.beta(2, 2, (n, 6)))
    prev = np.concatenate([bill[:, :1], bill[:, :-1]], axis=1)
    pamt = np.round(prev * frac)
    bill = np.round(bill)

    latest, mx = np.maximum(status[:, -1], 0), np.maximum(status, 0).max(1)
    logit = -1.9 + 0.9 * (latest >= 1) + 0.25 * np.minimum(mx, 4) + 0.2 * risk
    y = (rng.random(n) < sig(logit)).astype(int)

    df = pd.DataFrame({"id": np.arange(1, n + 1), "limit_bal": limit,
                       "sex": rng.choice([1, 2], n), "education": rng.choice([1, 2, 3, 4], n, p=[.35, .47, .16, .02]),
                       "marriage": rng.choice([1, 2, 3], n, p=[.45, .53, .02]), "age": rng.integers(21, 70, n)})
    for k in range(1, 7):
        df[f"pay_{k}"] = status[:, 6 - k]
    for k in range(1, 7):
        df[f"bill_amt{k}"] = bill[:, 6 - k]
    for k in range(1, 7):
        df[f"pay_amt{k}"] = pamt[:, 6 - k]
    df[TARGET] = y

    # dirty it up like the real file
    bad = rng.choice(n, int(.015 * n), replace=False)
    df.loc[bad, "education"] = rng.choice([0, 5, 6], len(bad))
    df.loc[rng.choice(n, max(3, int(.002 * n)), replace=False), "marriage"] = 0
    df.loc[rng.choice(n, int(.01 * n), replace=False), "bill_amt1"] *= -1
    dups = df.sample(10, random_state=seed).copy()
    dups["id"] = np.arange(n + 1, n + 11)
    return pd.concat([df, dups], ignore_index=True)[RAW_COLUMNS]
