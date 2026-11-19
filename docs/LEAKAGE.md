# Leakage risks checked

Each item names the risk, how it is controlled, and where to see the evidence.

| # | Risk | Control | Evidence |
|---|---|---|---|
| 1 | **Target leakage in features.** A feature that encodes the label or the future. | All features come from the 6 months of status, bill and payment history. The target is *next* month's default. `id` is only the row index; the target is never in `X`. A univariate-AUC scan flags any feature above 0.85 or below 0.15 (diagnostic only, never used to select features). | `leakage_checks.py`; `tests/test_pipeline_leakage.py::test_feature_matrix_has_no_target_or_id`, `::test_no_single_feature_is_a_label_proxy` |
| 2 | **Scaling / imputation / encoding before the split.** Fitting a scaler or imputer on all rows lets validation rows influence training. | `StandardScaler`, `SimpleImputer`, `OneHotEncoder` live inside the `ColumnTransformer` inside the `Pipeline`, so each CV fold refits them on its training rows. The only pre-model transforms (signed-log of amounts, and the clipping of repayment statuses used by `lr_binned`) are stateless. | `pipelines.py`; test `::test_scaler_is_fitted_on_training_rows_only` (fitted mean equals the train-fold mean, not the full-data mean) and `::test_validation_rows_do_not_influence_fit` |
| 3 | **Resampling before the split.** Oversampling then splitting puts synthetic copies of validation rows in training. | SMOTE and random undersampling are steps of an `imblearn` Pipeline: applied at `fit` on the training fold only, skipped at `predict`. | `make_pipeline`; test `::test_resampling_only_applies_at_fit_time` |
| 4 | **Class prior leaking through weights.** | `class_weight='balanced'` is computed inside `fit` from the training fold; XGBoost's `scale_pos_weight` is computed in `BalancedXGB.fit`, not from the full data. | `pipelines.py` |
| 5 | **Tuning on the evaluation folds.** | Nested CV: the inner `RandomizedSearchCV` sees only the outer-training rows; the outer test fold is scored once. | `run_experiments.nested_cv` |
| 6 | **Threshold chosen on the test fold.** | The threshold is picked from inner out-of-fold predictions on the outer-*training* rows, then applied to the test fold. Pooled confusion matrices use each fold's own threshold. | `run_experiments.nested_cv`; the cost-ratio sensitivity (`extra_analyses.refit_fold_costs`) re-selects thresholds the same way, and `test_refit_reproduces_nested_cv_thresholds_and_costs` checks it reproduces the main run |
| 7 | **Duplicate customers across folds.** | Exact duplicate rows (all columns except `id`) are dropped before any split. The id is unique, so there is one row per customer and no grouping is needed. | `cleaning.drop_exact_duplicates`, `sql/02_clean_views.sql` |
| 8 | **SQL feature engineering using cross-customer statistics.** | Window functions partition by customer `id` only; no global mean, quantile or encoding is computed in SQL. | `sql/03_features.sql`; test `test_features_are_row_local_no_cross_customer_statistics` |
| 9 | **Explanations or calibration computed on training fits.** | Final refits (for SHAP and coefficients) are used only for explanation. Performance and the calibration plot use out-of-fold predictions. | `interpret.py`, `plots.plot_calibration` |

## Residual risks (not eliminated)
- The SHAP analysis explains a final model refit on all data (a 3,000-customer sample); it is used for explanation only, never for any reported performance number.
- The hyperparameters shown for the final models come from a search on all data, so they are not an out-of-sample estimate.
- The file gives no exact observation date. We assume the Sept status and bill are known before the October outcome. If the status is recorded after the outcome window opens, `pay_1` would leak.
- Fixed cleaning rules (education/marriage recoding) were chosen after looking at the whole file. They are label-free, so the risk is small.
- Folds are random, not temporal (see README limitations).
