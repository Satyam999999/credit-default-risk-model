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
make cv           # nested CV, 5 models  -> results/cv_*.csv, oof_predictions.csv, figures/
make interpret    # coefficients, importances, SHAP -> results/, figures/
make report       # reports/comparison_report.md
```
No database? Use `--source file` (or `--source file --path data/<uci file>`) on any command; features then come from the pandas mirror.
`--fast` shrinks the search for a quick check (e.g. `python -m creditrisk.run_experiments --source file --models lr,lr_binned --fast --out /tmp/check`). Full `make cv` runs 5 models x 5 outer folds x 20 search iterations x 3 inner folds, so expect from tens of minutes to a few hours depending on cores. Lower `--n-iter` to trade search depth for time.

## Design choices (and assumptions)

- **SQL features** (per customer, window functions over that customer's 6 months): average and trend (`REGR_SLOPE`) of payment-to-bill ratio, last-3-month ratio, months delinquent (all / last 3), maximum delay, delay trend, average / latest / trend of utilization.
- **Cost assumption (not data-derived):** a missed default costs 5x a false alarm (`config.COST_FN/COST_FP`; `--c-fn/--c-fp`). The threshold minimises this cost, chosen on training data only. Change it to match real economics.
- **Evaluation:** nested stratified CV. The intervals are t-intervals across outer folds, which are optimistic because folds share data. Paired differences against the baseline use the same folds.
- **Resampling** uses `imbalanced-learn`'s Pipeline (a subclass of scikit-learn's), because scikit-learn's own Pipeline cannot resample.
- Hyperparameter search is `RandomizedSearchCV` (not Optuna), to stay scikit-learn-native.

## Results

Results from the full-settings nested cross-validation (5 outer folds, 3 inner folds, 20 randomized search iterations per fold, train-only cost-based threshold tuning with missed default = 5, false alarm = 1). Intervals are 95% fold-based intervals:

### Cross-Validation Summary (`results/cv_summary.csv`)

| Model | ROC-AUC | PR-AUC | Recall | Precision | Cost per customer |
|---|---|---|---|---|---|
| **Logistic regression (`lr`)** | 0.767 (0.751 to 0.784) | 0.522 (0.505 to 0.539) | 0.744 (0.712 to 0.776) | 0.369 (0.348 to 0.391) | 0.565 (0.539 to 0.591) |
| **Binned scorecard (`lr_binned`)** | 0.778 (0.768 to 0.788) | 0.547 (0.534 to 0.560) | 0.746 (0.720 to 0.771) | 0.383 (0.365 to 0.400) | 0.548 (0.524 to 0.572) |
| **Random forest (`rf`)** | 0.786 (0.774 to 0.798) | 0.561 (0.546 to 0.576) | 0.747 (0.703 to 0.791) | 0.383 (0.345 to 0.421) | 0.550 (0.532 to 0.568) |
| **XGBoost (`xgb`)** | 0.788 (0.776 to 0.799) | 0.565 (0.548 to 0.581) | 0.765 (0.733 to 0.796) | 0.373 (0.349 to 0.397) | 0.547 (0.526 to 0.568) |
| **LightGBM (`lgbm`)** | 0.788 (0.776 to 0.799) | 0.566 (0.548 to 0.584) | 0.764 (0.733 to 0.796) | 0.373 (0.338 to 0.408) | 0.548 (0.523 to 0.573) |

### Paired Differences vs Binned Scorecard (`lr_binned`)

Paired fold differences ($Model - lr\_binned$). Cost differences are **not detectable** (all 95% intervals cross zero), while tree models maintain a small, consistent ranking edge:

| Challenger vs `lr_binned` | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |
|---|---|---|---|---|
| **Random forest** | +0.0083 (0.0041 to 0.0124) | +0.0138 (0.0055 to 0.0221) | +0.0012 (-0.0524 to 0.0548) | +0.0017 (-0.0085 to 0.0119) |
| **XGBoost** | +0.0096 (0.0052 to 0.0140) | +0.0173 (0.0095 to 0.0250) | +0.0192 (-0.0336 to 0.0719) | -0.0016 (-0.0159 to 0.0128) |
| **LightGBM** | +0.0097 (0.0056 to 0.0138) | +0.0187 (0.0091 to 0.0282) | +0.0186 (-0.0272 to 0.0643) | 0.0000 (-0.0146 to 0.0146) |

### Operational Ranking (Defaults Captured in Top Deciles)

Out-of-fold ranking evaluated against actual defaults (6,630 total defaults across 29,965 customers; exactly 1,326 defaults per 5,993-customer fold):

- **Top 10% Risk Scores**:
  - Defaults captured: `lr_binned` = 416.0 defaults/fold; `lgbm` = 426.2 defaults/fold. Difference: **+10.2 defaults** (95% CI: **[+2.7, +17.7 defaults]**, excludes 0).
  - Capture percentage: `lr_binned` = 31.37%; `lgbm` = 32.14%. Difference: **+0.77%** (95% CI: **[+0.20%, +1.34%]**, excludes 0).
- **Top 20% Risk Scores**:
  - Defaults captured: `lr_binned` = 677.2 defaults/fold; `lgbm` = 685.4 defaults/fold. Difference: **+8.2 defaults** (95% CI: **[-2.3, +18.7 defaults]**, includes 0).
  - Capture percentage: `lr_binned` = 51.07%; `lgbm` = 51.69%. Difference: **+0.62%** (95% CI: **[-0.17%, +1.41%]**, includes 0).
- **Top 30% Risk Scores**:
  - Defaults captured: `lr_binned` = 833.4 defaults/fold; `lgbm` = 845.4 defaults/fold. Difference: **+12.0 defaults** (95% CI: **[+5.5, +18.5 defaults]**, excludes 0).
  - Capture percentage: `lr_binned` = 62.85%; `lgbm` = 63.76%. Difference: **+0.90%** (95% CI: **[+0.42%, +1.39%]**, excludes 0).

### Cost-Ratio Sensitivity (`results/cost_sensitivity.csv`)

Train-only threshold selection re-optimized independently at FN:FP ratios 2, 5, and 10 across all 5 folds:
- **Ratio 5:1 (primary)**: Cost differences between tree models and `lr_binned` are **not detectable** (all 95% CIs include zero; e.g. `lgbm` vs `lr_binned` diff: 0.0000 [-0.0146, +0.0146]). The gain from binning appears at 5:1, where linear `lr` incurs an extra +0.0168 cost penalty ([+0.0083, +0.0254]).
- **Ratio 2:1 (low penalty)**: Tree ensembles are slightly better and never worse (-0.006 cost units, detectable, 95% CI: [-0.010, -0.002]). The gain from binning does not appear at 2:1 (linear `lr` vs `lr_binned` difference is -0.0003, CI includes zero), showing binning gains emerge as default penalties rise.
- **Ratio 10:1 (high penalty)**: Tree models show lower average point-estimate cost (~ -0.021 units), but fold variance is wide and the difference is not detectable (95% CI: [-0.050, +0.008]).

### Interpretability & SHAP Grouping (`results/shap_summary.csv`)

SHAP values are calculated using `TreeExplainer` on a final LightGBM model refit on **all** training data (explaining a fixed 3,000-customer sample; not out-of-fold).
In transformed-feature terms (39 features total after one-hot encoding categorical variables):
- **Delinquency features (10 features)**: Account for **43.6% (~44%)** of total importance (summed mean $|SHAP| = 0.765$). Recent delinquency (`pay_1`) is the largest single driver (mean $|SHAP| = 0.251$), followed by `n_delinquent_last3` (0.184), `n_months_delinquent` (0.144), and `max_delay` (0.129).
- **Financial & demographic features (29 transformed features)**: Account for **56.4%** of total importance (summed mean $|SHAP| = 0.990$), led by `limit_bal` (0.099) and payment amounts.
- Final LightGBM hyperparameters selected via RandomizedSearchCV:
  `colsample_bytree = 0.598`, `learning_rate = 0.011`, `min_child_samples = 59`, `n_estimators = 362`, `num_leaves = 30`, `reg_lambda = 5.243`, `subsample = 0.816`.

## Limitations

- **Dataset scope**: One dataset from one bank in Taiwan (2005); results may not generalize across different economic environments or credit products.
- **Validation structure**: Stratified cross-validation, not time-based out-of-time (OOT) validation. Performance under macroeconomic stress is untested.
- **Fairness & regulation**: Age, sex, and marital status are included in the feature set; in many jurisdictions, their inclusion is legally restricted.
- **Cost ratio assumption**: Abstract cost ratio (FN=5, FP=1) is assumed; real charge-off losses and customer intervention costs vary by exposure and recovery rate.
- **Monthly granularity**: Delay codes approximate days-past-due at monthly increments. See `docs/LEAKAGE.md` for residual risks.

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
