# Data-quality findings (generated)

| Check | Rows | % of rows | Treatment |
|---|---:|---:|---|
| total rows | 30000 | 100.0 |  |
| default rate (target = 1) | 6636 | 22.12 | Class imbalance, see imbalance study. |
| education code outside documented 1-4 | 345 | 1.15 | Recoded to 4 (other); fixed rule. |
| marriage code outside documented 1-3 | 54 | 0.18 | Recoded to 3 (other); fixed rule. |
| rows with any pay status of -2 (undocumented) | 6561 | 21.87 | Kept; treated as 'no delay' (max(status,0)). Commonly read as 'no consumption'. |
| rows with any pay status of 0 (undocumented) | 21173 | 70.577 | Kept; treated as 'no delay' (revolving credit / minimum paid). |
| rows with any pay status >= 8 (very long delay) | 25 | 0.083 | Kept; plausible but rare, trees are robust, linear model sees delay as numeric. |
| exact duplicate rows (ignoring id, excluding first copy) | 35 | 0.117 | Dropped before any split to avoid train/validation leakage. |
| rows with a negative bill amount | 1930 | 6.433 | Kept (credit balance); utilization uses max(bill, 0). |
| rows with a bill above 1.5x the credit limit | 343 | 1.143 | Kept (over-limit is possible); linear model uses signed-log amounts. |
| rows with a payment above the credit limit | 280 | 0.933 | Kept; payment/bill ratio is capped at 1. |
| rows with a delay status but no positive previous bill | 1531 | 5.103 | Kept (inconsistent: nothing owed yet flagged late); ratio is undefined -> missing. |
| rows with status 0 but zero payment against a positive previous bill | 2548 | 8.493 | Kept (status 0 should mean minimum paid); the ratio feature exposes it to the model. |
| rows where delay jumps from <=0 to >=2 in one month | 5948 | 19.827 | Kept (logically a delay can grow by at most 1 per month); flagged as label noise. |
| rows with age outside 21-79 | 0 | 0.0 | Kept. |
