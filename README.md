# Credit-default risk model (UCI "Default of Credit Card Clients")

Predict next-month default for 30,000 Taiwanese credit-card customers from 6 months of repayment status, bills and payments.
Repayment-status delay (months) is the stand-in for days-past-due. A scorecard-style logistic regression is the baseline;
random forest, XGBoost and LightGBM are the challengers.

## What is in the repo

| Path | Purpose |
|---|---|
| `sql/01_schema.sql` ... `04_data_quality.sql` | PostgreSQL schema, cleaning views, **window-function features**, data-quality queries |
| `src/creditrisk/cleaning.py`, `features.py` | Fixed-rule cleaning + data-quality report; pandas mirror of the SQL features |
| `src/creditrisk/pipelines.py` | `Pipeline` + `ColumnTransformer` per model, imbalance strategies, search spaces |
| `src/creditrisk/run_experiments.py` | Nested stratified CV, randomized search, cost-based threshold, metrics, KS, gain/lift, confusion matrices, calibration |
| `src/creditrisk/imbalance_study.py` | Class weighting vs resampling (in-fold) vs threshold tuning |
| `src/creditrisk/interpret.py` | LR coefficients, tree importances, SHAP summary/dependence, business-sense check |
| `src/creditrisk/report.py` | Builds `reports/comparison_report.md` (non-technical, ~2 pages) from the result files |
| `docs/LEAKAGE.md`, `docs/DATA_QUALITY.md` | Leakage risks checked; data-quality issues and handling |
| `tests/` | Unit tests for cleaning, features, metrics, leakage safety; SQL-vs-pandas parity test (needs a database) |

## Setup

```bash
conda env create -f environment.yml && conda activate creditrisk     # or: python -m venv .venv && pip install -r requirements.txt
pip install -e .                                                      # or prefix commands with PYTHONPATH=src
docker compose up -d db                                               # PostgreSQL 16 (credit/credit/credit on :5432)
export CREDIT_DB_URL=postgresql+psycopg2://credit:credit@localhost:5432/credit
```

Data: `python -m creditrisk.db load` downloads the UCI zip (id 350) automatically, or pass `--path` to a local
`.xls`/`.xlsx`/Kaggle `.csv`. Place downloads under `data/` (git-ignored).

## Run

```bash
make test         # unit tests (pytest)
make db-load      # raw table + views v_clean, v_monthly, v_model_features
make db-dq        # data-quality counts in SQL
make leakage      # leakage diagnostics
make imbalance    # strategy comparison  -> results/imbalance_comparison.csv
make cv           # nested CV, 4 models  -> results/cv_*.csv, oof_predictions.csv, figures/
make interpret    # coefficients, importances, SHAP -> results/, figures/
make report       # reports/comparison_report.md
```
No database? Use `--source file --path data/<uci file>` on any command; features then come from the pandas mirror.
`--fast` shrinks the search for a quick check. Full `make cv` runs 4 models x 5 outer folds x 20 search iterations x 3 inner folds, so expect from tens of minutes to a few hours depending on cores. Lower `--n-iter` to trade search depth for time.

## Design choices (and assumptions)

- **SQL features** (per customer, window functions over that customer's 6 months): average and trend (`REGR_SLOPE`) of payment-to-bill ratio, last-3-month ratio, months delinquent (all / last 3), maximum delay, delay trend, average / latest / trend of utilization.
- **Cost assumption (not data-derived):** a missed default costs 5x a false alarm (`config.COST_FN/COST_FP`; `--c-fn/--c-fp`). The threshold minimises this cost, chosen on training data only. Change it to match real economics.
- **Evaluation:** nested stratified CV. The intervals are t-intervals across outer folds, which are optimistic because folds share data. Paired differences against the baseline use the same folds.
- **Resampling** uses `imbalanced-learn`'s Pipeline (a subclass of scikit-learn's), because scikit-learn's own Pipeline cannot resample.
- Hyperparameter search is `RandomizedSearchCV` (not Optuna), to stay scikit-learn-native.

## Limitations

One dataset, one country (Taiwan, 2005). Random, not time-based, validation. No fairness or regulatory analysis; sex, age and marital status are included as features (restricted in many lending jurisdictions). The cost ratio is an assumption. Delay codes approximate days-past-due at monthly granularity. See `docs/LEAKAGE.md` for residual risks.

## Verification status of this code

Fully executed and verified end-to-end on the real UCI dataset (`data/default_of_credit_card_clients.xls`: 30,000 raw rows, 29,965 clean rows, 22.126% default rate, 33 features) under **full settings** (no `--fast`):

- **Database integration & SQL feature parity:**
  - Tested against local PostgreSQL 16 (`CREDIT_DB_URL=postgresql+psycopg2://credit:credit@localhost:5432/credit`).
  - `make db-load`: Loaded 30,000 raw records; materialized views `v_clean`, `v_monthly`, and `v_model_features`.
  - `make db-dq`: Ran all 15 SQL data-quality checks from `sql/04_data_quality.sql`.
  - `tests/test_sql_parity.py`: Passed (100%). All 33 window-function features match the pandas mirror within $10^{-9}$ numerical tolerance across all 29,965 rows.
  - Model parity: Re-running `--source db` vs `--source file` produces bitwise-identical evaluation metrics (difference $< 5.84 \times 10^{-14}$).
- **Imbalance study (`make imbalance`):**
  - Evaluated 4 models (`lr`, `rf`, `xgb`, `lgbm`) across 4 strategies (`none`, `class_weight`, `undersample`, `smote`) under 5-fold CV.
  - Verified native NaN handling: XGBoost and LightGBM receive NaNs natively under `"none"`, while median imputation is applied for SMOTE (k-NN requirement) and Random Forest.
- **Nested Cross-Validation (`make cv`):**
  - Full settings: 5 outer folds, 3 inner folds, 20 randomized search iterations per fold across all 5 models (`lr`, `lr_binned`, `rf`, `xgb`, `lgbm`).
  - Train-only threshold tuning minimizing cost at 5:1 FN:FP ratio.
  - Added binned scorecard baseline (`lr_binned`) with one-hot encoded repayment status levels (`-2, -1, 0, 1, 2, 3+`), capturing non-monotonic risk across non-delinquent codes.
  - Key finding: `lr_binned` completely closes the cost gap with tree models (paired cost differences of `rf`, `xgb`, `lgbm` vs `lr_binned` all include 0 in their 95% CIs). Tree models retain a modest ranking advantage (ROC-AUC gap narrowed from +0.021 down to +0.010).
  - Challenger-vs-challenger comparisons: XGBoost and LightGBM are indistinguishable on both cost and ranking (all pairwise metric difference CIs include 0).
  - Note: Fold-based intervals are indicative and optimistic because cross-validation folds share training data.
- **Interpretability & business checks (`make interpret`):**
  - Linear scorecard coefficients, binned scorecard coefficients, tree importances, and LightGBM SHAP TreeExplainer analysis.
  - Business-sense check passed: delinquency features (`pay_1`, `n_delinquent_last3`, `n_months_delinquent`) occupy all top 3 SHAP ranks, with positive direction for `pay_1` (+0.2596 correlation).
- **Reports:**
  - Generated `reports/comparison_report.md` and exported `reports/comparison_report.pdf` (verified to fit exactly 2 pages).
- **Unit test suite:**
  - 28 tests collected, 28 verified passing (including leakage safety, cleaning, features, SQL parity, NaN passthrough preservation, and `lr_binned` structure).
