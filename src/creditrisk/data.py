"""Dataset access: UCI file, PostgreSQL feature view, or synthetic (tests only)."""
from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaning import (clean_categories, data_quality_report, drop_exact_duplicates, standardize_columns)
from .config import DATA_DIR, FEATURES, TARGET, UCI_URL
from .features import add_engineered_features


@dataclass
class Dataset:
    X: pd.DataFrame
    y: pd.Series
    n_raw: int
    n_clean: int
    source: str


def download_uci(dest: Path = DATA_DIR) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(UCI_URL, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith((".xls", ".xlsx", ".csv")))
        z.extract(name, dest)
    return dest / name


def read_raw_file(path: str | Path | None = None) -> pd.DataFrame:
    """Read the UCI file (.xls/.xlsx has a junk first row; Kaggle-style .csv does not)."""
    if path:
        p = Path(path)
    else:
        # Look for local dataset in DATA_DIR before attempting network download
        local_files = [f for f in DATA_DIR.iterdir() if f.is_file() and f.suffix.lower() in (".xls", ".xlsx", ".csv")] if DATA_DIR.exists() else []
        if local_files:
            # Sort by file size descending to pick real dataset file over tiny stubs
            local_files.sort(key=lambda f: f.stat().st_size, reverse=True)
            p = local_files[0]
        else:
            p = download_uci()
    reader = (lambda h: pd.read_excel(p, header=h)) if p.suffix.lower() in (".xls", ".xlsx") \
        else (lambda h: pd.read_csv(p, header=h))
    df = reader(0)
    if "id" not in [str(c).strip().lower() for c in df.columns]:
        df = reader(1)
    return standardize_columns(df)


def prepare_from_raw(raw: pd.DataFrame, source: str = "file") -> Dataset:
    clean = clean_categories(drop_exact_duplicates(raw))
    feats = add_engineered_features(clean).set_index("id")
    return Dataset(feats[FEATURES], feats[TARGET].astype(int), len(raw), len(clean), source)


def load_dataset(source: str = "file", path: str | None = None, db_url: str | None = None,
                 n_synthetic: int = 4000) -> Dataset:
    if source == "synthetic":
        from .synthetic import make_synthetic
        print("!! SYNTHETIC DATA: pipeline smoke test only. Do NOT report these numbers. !!")
        return prepare_from_raw(make_synthetic(n_synthetic), "synthetic")
    if source == "db":
        from .db import get_engine, read_features
        df = read_features(get_engine(db_url)).set_index("id")
        return Dataset(df[FEATURES], df[TARGET].astype(int), len(df), len(df), "db")
    return prepare_from_raw(read_raw_file(path), "file")


def write_dq_report(raw: pd.DataFrame, path: Path) -> pd.DataFrame:
    rep = data_quality_report(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Data-quality findings (generated)", "",
             "| Check | Rows | % of rows | Treatment |", "|---|---:|---:|---|"]
    lines += [f"| {r.check} | {r.n_rows} | {r.pct_of_rows} | {r.treatment} |" for r in rep.itertuples()]
    path.write_text("\n".join(lines) + "\n")
    return rep
