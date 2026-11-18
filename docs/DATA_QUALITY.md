# Data-quality issues

Counts are produced by `python -m creditrisk.run_experiments --source file` (writes `docs/DATA_QUALITY_FINDINGS.md`)
and by `sql/04_data_quality.sql` (run `make db-dq`). Run them on the real file; this page describes the checks and how each is handled.

Conventions: `pay_1`/`bill_amt1`/`pay_amt1` are the most recent month (Sept 2005). The UCI file names `pay_1` as `PAY_0`.
The payment in month k is treated as paying the statement of the older month k+1.

| Issue | Why it matters | Handling |
|---|---|---|
| EDUCATION codes 0, 5, 6 (documented 1-4) | Undocumented categories | Recoded to 4 (other) |
| MARRIAGE code 0 (documented 1-3) | Undocumented category | Recoded to 3 (other) |
| PAY status values -2 and 0 (documented only -1 and 1-9) | Meaning not given in the documentation; commonly read as -2 = no consumption, 0 = revolving / minimum paid | Treated as "no delay": `delay_months = max(status, 0)`. Raw statuses stay as model inputs |
| Exact duplicate rows | Same record in train and validation inflates scores | Dropped before splitting |
| Negative bill amounts | Credit balances | Kept; utilization uses `max(bill, 0)` |
| Bills above the limit; payments above the limit | Outliers / over-limit | Kept; ratio capped at 1; linear model uses signed-log amounts; tree models are insensitive to scale |
| Delay flagged but previous bill <= 0 | Inconsistent: nothing owed yet "late" | Kept; pay ratio undefined (NULL) |
| Status 0 but zero payment against a positive bill | Status 0 should mean the minimum was paid | Kept; the ratio feature exposes it |
| Delay jumping from <=0 to >=2 in one month | A delay can grow by at most 1 per month | Kept; flagged as likely label noise |
| Age outside 21-79, heavy-tailed amounts | Outliers | Kept |
| Class imbalance (about 22% default in the published dataset) | Affects thresholds | See imbalance study |

Imputation for missing engineered values (undefined ratios or slopes) happens inside the pipeline, fitted per fold.
