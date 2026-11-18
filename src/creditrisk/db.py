"""PostgreSQL helpers: create schema, bulk-load raw data, build SQL views, read features.

    python -m creditrisk.db load --path data/default_of_credit_card_clients.xls
    python -m creditrisk.db dq
"""
from __future__ import annotations

import argparse
import io

import pandas as pd

from .config import DB_URL, RAW_COLUMNS, SQL_DIR


def get_engine(url: str | None = None):
    from sqlalchemy import create_engine
    return create_engine(url or DB_URL)


def run_sql_file(engine, name: str) -> None:
    sql = (SQL_DIR / name).read_text()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)


def load_raw(engine, raw: pd.DataFrame) -> int:
    """Create schema and COPY the standardized raw frame into raw_credit."""
    run_sql_file(engine, "01_schema.sql")
    buf = io.StringIO()
    raw[RAW_COLUMNS].to_csv(buf, index=False, header=False)
    buf.seek(0)
    conn = engine.raw_connection()
    try:
        with conn.cursor() as cur:
            cur.copy_expert(f"COPY raw_credit ({', '.join(RAW_COLUMNS)}) FROM STDIN WITH CSV", buf)
        conn.commit()
    finally:
        conn.close()
    run_sql_file(engine, "02_clean_views.sql")
    run_sql_file(engine, "03_features.sql")
    return len(raw)


def read_features(engine) -> pd.DataFrame:
    return pd.read_sql("SELECT * FROM v_model_features ORDER BY id", engine)


def data_quality_sql(engine) -> pd.DataFrame:
    return pd.read_sql((SQL_DIR / "04_data_quality.sql").read_text(), engine)


def main() -> None:
    from .data import read_raw_file
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["load", "dq"])
    ap.add_argument("--path", default=None, help="UCI .xls/.xlsx/.csv (downloaded if omitted)")
    ap.add_argument("--db-url", default=None)
    a = ap.parse_args()
    eng = get_engine(a.db_url)
    if a.action == "load":
        n = load_raw(eng, read_raw_file(a.path))
        print(f"Loaded {n} rows into raw_credit and created views v_clean, v_monthly, v_model_features.")
    else:
        print(data_quality_sql(eng).to_string(index=False))


if __name__ == "__main__":
    main()
