"""Central configuration: paths, column groups, and the (assumed) cost matrix."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
SQL_DIR = ROOT / "sql"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
REPORTS_DIR = ROOT / "reports"
DOCS_DIR = ROOT / "docs"

SEED = 42
TARGET = "default_next_month"

# UCI dataset id 350 (Yeh & Lien 2009). The download is fetched automatically if no local file is found.
UCI_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
DB_URL = os.environ.get("CREDIT_DB_URL", "postgresql+psycopg2://credit:credit@localhost:5432/credit")

# ---------------------------------------------------------------------------
# COST ASSUMPTION (not data-derived): a missed default costs 5x a false alarm.
# Think of the missed default as an unrecovered balance / charge-off and the
# false alarm as a limit review or outreach call. Change these to match the
# real economics; every threshold and "cost per customer" number follows.
# ---------------------------------------------------------------------------
COST_FN = 5.0
COST_FP = 1.0

# ---------------------------------------------------------------------------
# Columns. UCI numbering: *_1 is the MOST RECENT month (September 2005),
# *_6 the oldest (April 2005). The UCI file calls PAY_1 "PAY_0"; we rename it.
# ---------------------------------------------------------------------------
PAY = [f"pay_{k}" for k in range(1, 7)]
BILL = [f"bill_amt{k}" for k in range(1, 7)]
PAYAMT = [f"pay_amt{k}" for k in range(1, 7)]
RAW_COLUMNS = ["id", "limit_bal", "sex", "education", "marriage", "age"] + PAY + BILL + PAYAMT + [TARGET]

ENGINEERED = [
    "avg_pay_ratio", "pay_ratio_trend", "pay_ratio_last3",
    "n_months_delinquent", "n_delinquent_last3", "max_delay", "delay_trend",
    "avg_utilization", "utilization_latest", "utilization_trend",
]
CATEGORICAL = ["sex", "education", "marriage"]
AMOUNT_COLS = ["limit_bal"] + BILL + PAYAMT          # heavy-tailed, signed-log for linear model
OTHER_NUMERIC = ["age"] + PAY + ENGINEERED
FEATURES = CATEGORICAL + AMOUNT_COLS + OTHER_NUMERIC

DELINQUENCY_FEATURES = ["pay_1", "pay_2", "n_months_delinquent", "n_delinquent_last3", "max_delay", "delay_trend"]
# All features that describe repayment delay; used for the grouped SHAP importance.
DELINQUENCY_GROUP = PAY + ["n_months_delinquent", "n_delinquent_last3", "max_delay", "delay_trend"]

MODEL_LABELS = {
    "lr": "Logistic regression (baseline)",
    "lr_binned": "Binned logistic regression (scorecard)",
    "rf": "Random forest",
    "xgb": "XGBoost",
    "lgbm": "LightGBM",
}

FEATURE_DESCRIPTIONS = {
    "pay_1": "repayment status last month (delay in months)",
    "pay_2": "repayment status two months ago",
    "pay_3": "repayment status three months ago",
    "n_months_delinquent": "number of the last 6 months with a payment delay",
    "n_delinquent_last3": "number of the last 3 months with a payment delay",
    "max_delay": "worst payment delay (months) in the last 6 months",
    "delay_trend": "whether delays are getting worse or better over time",
    "avg_pay_ratio": "average share of the previous bill that was paid",
    "pay_ratio_trend": "whether the share of the bill paid is rising or falling",
    "pay_ratio_last3": "share of the previous bill paid in the last 3 months",
    "avg_utilization": "average share of the credit limit in use",
    "utilization_latest": "share of the credit limit in use last month",
    "utilization_trend": "whether credit use is rising or falling",
    "limit_bal": "credit limit",
    "age": "customer age",
}
for _k in range(1, 7):
    FEATURE_DESCRIPTIONS.setdefault(f"bill_amt{_k}", f"bill amount, month {_k} (1 = most recent)")
    FEATURE_DESCRIPTIONS.setdefault(f"pay_amt{_k}", f"payment amount, month {_k} (1 = most recent)")
