# Credit-default risk model (UCI "Default of Credit Card Clients")

[![Tests](https://github.com/Satyam999999/credit-default-risk-model/actions/workflows/tests.yml/badge.svg)](https://github.com/Satyam999999/credit-default-risk-model/actions/workflows/tests.yml)

A binned logistic-regression scorecard shows no detectable cost difference from boosted trees at an assumed 5:1 missed-default:false-alarm cost, while tree models keep a small ranking edge (ROC-AUC about +0.010). Fold-based intervals are optimistic; no time-based validation.

Predict next-month default for 30,000 Taiwanese credit-card customers from 6 months of repayment status, bills and payments.
Repayment-status delay (months) is the stand-in for days-past-due. There are two baselines: linear logistic regression (`lr`)
and binned logistic regression (`lr_binned`, which treats repayment statuses as categorical bins and is closer to a real scorecard);
random forest (`rf`), XGBoost (`xgb`) and LightGBM (`lgbm`) are the challengers.

[Full Report (2-page PDF)](reports/comparison_report.pdf)

![Gain and lift](results/figures/gain_lift.png)
![Calibration](results/figures/calibration.png)

## What is in the repo

| Path | Purpose |
|---|---|
| `sql/01_schema.sql` ... `04_data_quality.sql` | PostgreSQL schema, cleaning views, **window-function features**, data-quality queries |
| `src/creditrisk/cleaning.py`, `features.py` | Fixed-rule cleaning + data-quality report; pandas mirror of the SQL features |
| `src/creditrisk/pipelines.py` | `Pipeline` + `ColumnTransformer` per model, imbalance strategies, search spaces |
| `src/creditrisk/run_experiments.py` | Nested stratified CV, randomized search, cost-based threshold, metrics, KS, gain/lift, confusion matrices, calibration |
| `src/creditrisk/imbalance_study.py` | Class weighting vs resampling (in-fold) vs threshold tuning |
| `src/creditrisk/interpret.py` | LR coefficients, tree importances, SHAP summary/dependence, business-sense check |
| `src/creditrisk/extra_analyses.py` | Operational ranking (defaults captured in the top 10/20/30%), cost-ratio sensitivity with train-only thresholds, grouped SHAP importance |
| `src/creditrisk/report.py` | Builds `reports/comparison_report.md` (non-technical, ~2 pages) and the PDF; every number is computed from `results/` |
| `docs/LEAKAGE.md`, `docs/DATA_QUALITY.md` | Leakage risks checked; data-quality issues and handling |
| `docs/DATA_QUALITY_FINDINGS.md` | Generated data-quality counts and value distributions from the cleaning pipeline |
| `tests/` | Unit tests for cleaning, features, metrics, leakage safety; SQL-vs-pandas parity test (needs a database) |

Diagnostic figures live in `results/figures/` (`gain_lift.png`, `confusion_matrices.png`, `calibration.png`, `roc_pr.png`, `ks_*.png`), while KS statistics are reported per model in `results/cv_summary.csv`.

## Setup

```bash
conda env create -f environment.yml && conda activate creditrisk     # or: python -m venv .venv && pip install -r requirements.txt
pip install -e .                                                      # or prefix commands with PYTHONPATH=src
docker compose up -d db                                               # PostgreSQL 16 (credit/credit/credit on :5432)
export CREDIT_DB_URL=postgresql+psycopg2://credit:credit@localhost:5432/credit
```
*(Note: The Docker Compose path is untested; PostgreSQL 16 was verified locally via Homebrew.)*

Data: `python -m creditrisk.db load` downloads the UCI zip (id 350) automatically, or pass `--path` to a local
`.xls`/`.xlsx`/Kaggle `.csv`. Place downloads under `data/` (git-ignored).

## Run

```bash
make test         # unit tests (the SQL-vs-pandas parity test is skipped unless CREDIT_DB_URL is set)
make db-load      # raw table + views v_clean, v_monthly, v_model_features
make db-dq        # data-quality counts in SQL
make leakage      # leakage diagnostics
make imbalance    # strategy comparison  -> results/imbalance_comparison.csv
make cv           # nested CV, 5 models  -> results/cv_*.csv, oof_predictions.csv, figures/
make interpret    # coefficients, importances, SHAP -> results/, figures/
make analyze      # ranking, cost-ratio sensitivity, grouped SHAP -> results/operational_ranking*.csv, cost_sensitivity*.csv, shap_groups.json
make report       # reports/comparison_report.md (+ PDF if Chrome/Chromium and the `markdown` package are available)
```
`make all` runs the whole chain in the order test, leakage, imbalance, cv, interpret, analyze, report (it does not start Docker).
`make analyze` re-fits each model once per outer fold to re-select thresholds at other cost ratios; it prints a self-check against `results/cv_fold_metrics.csv` and flags any mismatch.
No database? Use `--source file` (or `--source file --path data/<uci file>`) on any command; features then come from the pandas mirror.
`--fast` shrinks the search for a quick check (e.g. `python -m creditrisk.run_experiments --source file --models lr,lr_binned --fast --out /tmp/check`). Full `make cv` runs 5 models x 5 outer folds x 20 search iterations x 3 inner folds, so expect from tens of minutes to a few hours depending on cores. Lower `--n-iter` to trade search depth for time.

## Design choices (and assumptions)

- **SQL features** (per customer, window functions over that customer's 6 months): average and trend (`REGR_SLOPE`) of payment-to-bill ratio, last-3-month ratio, months delinquent (all / last 3), maximum delay, delay trend, average / latest / trend of utilization.
- **Cost assumption (not data-derived):** a missed default costs 5x a false alarm (`config.COST_FN/COST_FP`; `--c-fn/--c-fp`). The threshold minimises this cost, chosen on training data only. Change it to match real economics.
- **Evaluation:** nested stratified CV. The intervals are t-intervals across outer folds, which are optimistic because folds share data. Paired differences against the baseline use the same folds.
- **Resampling** uses `imbalanced-learn`'s Pipeline (a subclass of scikit-learn's), because scikit-learn's own Pipeline cannot resample.
- Hyperparameter search is `RandomizedSearchCV` (not Optuna), to stay scikit-learn-native.

## Results

Results from the full-settings nested cross-validation (5 outer folds, 3 inner folds, 20 randomized search iterations per fold, train-only cost-based threshold tuning with missed default = 5, false alarm = 1). Intervals are 95% fold-based intervals. All numbers below are regenerated by `make cv`, `make interpret` and `make analyze`, and `reports/comparison_report.md` is built from the same files (a test checks that it matches).

### Cross-Validation Summary (`results/cv_summary.csv`)

| Model | ROC-AUC | PR-AUC | Recall | Precision | Cost per customer |
|---|---|---|---|---|---|
| **Logistic regression (`lr`)** | 0.767 (0.751 to 0.784) | 0.522 (0.505 to 0.539) | 0.744 (0.712 to 0.776) | 0.369 (0.348 to 0.391) | 0.565 (0.539 to 0.591) |
| **Binned scorecard (`lr_binned`)** | 0.778 (0.768 to 0.788) | 0.547 (0.534 to 0.560) | 0.746 (0.720 to 0.771) | 0.383 (0.365 to 0.400) | 0.548 (0.524 to 0.572) |
| **Random forest (`rf`)** | 0.786 (0.774 to 0.798) | 0.561 (0.546 to 0.576) | 0.747 (0.703 to 0.791) | 0.383 (0.345 to 0.421) | 0.550 (0.532 to 0.568) |
| **XGBoost (`xgb`)** | 0.788 (0.776 to 0.799) | 0.565 (0.548 to 0.581) | 0.765 (0.733 to 0.796) | 0.373 (0.349 to 0.397) | 0.547 (0.526 to 0.568) |
| **LightGBM (`lgbm`)** | 0.788 (0.776 to 0.799) | 0.566 (0.548 to 0.584) | 0.764 (0.733 to 0.796) | 0.373 (0.338 to 0.408) | 0.548 (0.523 to 0.573) |

### Paired Differences vs Binned Scorecard (`lr_binned`)

Paired fold differences (Model − lr_binned). Cost differences are **not detectable** (all 95% intervals cross zero), while tree models maintain a small, consistent ranking edge:

| Challenger vs `lr_binned` | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |
|---|---|---|---|---|
| **Random forest** | +0.0083 (0.0041 to 0.0124) | +0.0138 (0.0055 to 0.0221) | +0.0012 (-0.0524 to 0.0548) | +0.0017 (-0.0085 to 0.0119) |
| **XGBoost** | +0.0096 (0.0052 to 0.0140) | +0.0173 (0.0095 to 0.0250) | +0.0192 (-0.0336 to 0.0719) | -0.0016 (-0.0159 to 0.0128) |
| **LightGBM** | +0.0097 (0.0056 to 0.0138) | +0.0187 (0.0091 to 0.0282) | +0.0186 (-0.0272 to 0.0643) | 0.0000 (-0.0146 to 0.0146) |

### Operational Ranking (Defaults Captured in Top Deciles)

Out-of-fold ranking evaluated against actual defaults (6,630 total defaults across 29,965 customers; exactly 1,326 defaults per 5,993-customer fold). Cutoff rule: top-k% rounded to the nearest customer (1,199 for 20%):

- **Top 10% Risk Scores**:
  - Defaults captured: `lr_binned` = 416.0 defaults/fold; `lgbm` = 426.2 defaults/fold. Difference: **+10.2 defaults** (95% CI: **[+2.7, +17.7 defaults]**, excludes 0).
  - Capture percentage: `lr_binned` = 31.37%; `lgbm` = 32.14%. Difference: **+0.77 percentage points** (95% CI: **[+0.20, +1.34] pp**, excludes 0).
- **Top 20% Risk Scores**:
  - Defaults captured: `lr_binned` = 677.2 defaults/fold; `lgbm` = 685.4 defaults/fold. Difference: **+8.2 defaults** (95% CI: **[-2.3, +18.7 defaults]**, includes 0).
  - Capture percentage: `lr_binned` = 51.07%; `lgbm` = 51.69%. Difference: **+0.62 percentage points** (95% CI: **[-0.17, +1.41] pp**, includes 0).
- **Top 30% Risk Scores**:
  - Defaults captured: `lr_binned` = 833.4 defaults/fold; `lgbm` = 845.4 defaults/fold. Difference: **+12.0 defaults** (95% CI: **[+5.5, +18.5 defaults]**, excludes 0).
  - Capture percentage: `lr_binned` = 62.85%; `lgbm` = 63.76%. Difference: **+0.90 percentage points** (95% CI: **[+0.42, +1.39] pp**, excludes 0).

### Cost-Ratio Sensitivity (`results/cost_sensitivity.csv`)

Train-only threshold selection re-optimized independently at FN:FP ratios 2, 5, and 10 across all 5 folds:
- **Ratio 5:1 (primary)**: Cost differences between tree models and `lr_binned` are **not detectable** (all 95% CIs include zero; e.g. `lgbm` vs `lr_binned` diff: 0.0000 [-0.0146, +0.0146]).
- **Ratio 2:1 (low penalty)**: Tree ensembles are slightly better and never detectably worse (-0.006 cost units, detectable, 95% CI: [-0.010, -0.002]).
- **Ratio 10:1 (high penalty)**: Tree models show lower average point-estimate cost (~ -0.021 units), but fold variance is wide and the difference is not detectable (every model's 95% CI includes zero; e.g. LightGBM [-0.050, +0.008]).
- **Effect of binning**: Across ratios, binning shows no difference at 2:1, a detectable gain at 5:1 (+0.0168 linear penalty), and an undetermined effect at 10:1 (wide interval).

### Class Imbalance Strategies (`results/imbalance_comparison.csv`)

- At a 0.5 cut-off, class weighting or undersampling raises recall (~0.33-0.38 -> 0.61-0.66), lowers precision, and worsens Brier score (~0.13 -> 0.17-0.19).
- SMOTE leaves XGBoost and LightGBM close to no strategy; for random forest it raises recall (~0.37 -> ~0.49) and worsens Brier (~0.134 -> ~0.142).
- With a cost-tuned cut-off, all strategies cost 0.546-0.565, so we use no resampling and tune the threshold.
- *Note: `lr_binned` was not part of the imbalance study.*

### Interpretability & SHAP Grouping (`results/shap_summary.csv`)

SHAP values are calculated using `TreeExplainer` on a final LightGBM model refit on **all** training data (explaining a fixed 3,000-customer sample; not out-of-fold).
In transformed-feature terms (39 features total after one-hot encoding categorical variables):
- **Delinquency features (10 features)**: Account for **43.6% (~44%)** of total importance (summed mean `|SHAP| = 0.765`). Recent delinquency (`pay_1`) is the largest single driver (mean `|SHAP| = 0.251`), followed by `n_delinquent_last3` (0.184), `n_months_delinquent` (0.144), and `max_delay` (0.129).
- **Financial & demographic features (29 transformed features)**: Account for **56.4%** of total importance (summed mean `|SHAP| = 0.990`), led by `bill_amt1` (0.109), `limit_bal` (0.084), `utilization_latest` (0.082), and `pay_amt1` (0.074).
- Final LightGBM hyperparameters selected via RandomizedSearchCV:
  `colsample_bytree = 0.598`, `learning_rate = 0.011`, `min_child_samples = 59`, `n_estimators = 362`, `num_leaves = 30`, `reg_lambda = 5.243`, `subsample = 0.816`.

## Limitations

- **Dataset scope**: One dataset from one bank in Taiwan (2005); results may not generalize across different economic environments or credit products.
- **Validation structure**: Stratified cross-validation, not time-based out-of-time (OOT) validation. Performance under macroeconomic stress is untested.
- **Fairness & regulation**: Age, sex, and marital status are included in the feature set; in many jurisdictions, their inclusion is legally restricted.
- **Cost ratio assumption**: Abstract cost ratio (FN=5, FP=1) is assumed; real charge-off losses and customer intervention costs vary by exposure and recovery rate.
- **Monthly granularity**: Delay codes approximate days-past-due at monthly increments. See `docs/LEAKAGE.md` for residual risks.

## Testing

- `make test` runs 37 tests covering cleaning, features (including a hand-computed customer), metrics and thresholds, leakage safety (scalers fitted on training rows only, validation rows cannot influence a fit, resampling only at fit time, NaN handling per model), the cost-sensitivity refit (reproduces the main run's fold costs), and the report generator (the committed report equals a fresh build from `results/`).
- The SQL-vs-pandas parity test needs PostgreSQL and is skipped otherwise. It passed on PostgreSQL 16 (installed via Homebrew): all 33 features agree within 1e-9, and model metrics from the SQL and pandas paths agree to better than 1e-13.
- The leakage controls are tested, not proven; see `docs/LEAKAGE.md` for the checks and the residual risks.
- The Docker Compose path is untested. CI (GitHub Actions) runs the database-free tests on every push.

## Data and citation

Data: "Default of Credit Card Clients", UCI Machine Learning Repository (dataset id 350), downloaded on first use and not stored in this repository; see the UCI page for the dataset's terms.
Original study: Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. *Expert Systems with Applications*, 36(2), 2473-2480.

License: MIT
