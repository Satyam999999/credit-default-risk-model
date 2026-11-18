# Credit-default risk: comparison of scoring models

## Summary

We compared scorecard-style models (logistic regression and binned scorecard) with machine-learning alternatives on 29,965 credit-card customers, predicting next-month default. On the assumed cost of mistakes, **XGBoost**, **Binned logistic regression (scorecard)**, **Random forest** and **LightGBM** achieved comparable lowest average costs (0.547 to 0.550), with all mutual paired differences including zero (performance is indistinguishable within fold variation).

## What the baseline does

Logistic regression scores customers by adding up weighted signals (delinquency, payment ratios, utilization, limit, age). Each weight can be audited. The binned scorecard ('lr_binned') relaxes linearity by treating repayment statuses (-2 to 3+) as categorical dummy levels, capturing non-monotonic risk across non-delinquent codes.

## How the models compare

Average over 5 cross-validation folds with 95% fold-based intervals. Higher is better except cost.

| Model | ROC-AUC | PR-AUC | Recall | Precision | Cost per customer |
|---|---|---|---|---|---|
| Logistic regression (baseline) | 0.767 (0.751 to 0.784) | 0.522 (0.505 to 0.539) | 0.744 (0.712 to 0.776) | 0.369 (0.348 to 0.391) | 0.565 (0.539 to 0.591) |
| Binned logistic regression (scorecard) | 0.778 (0.768 to 0.788) | 0.547 (0.534 to 0.560) | 0.746 (0.720 to 0.771) | 0.383 (0.365 to 0.400) | 0.548 (0.524 to 0.572) |
| Random forest | 0.786 (0.774 to 0.798) | 0.561 (0.546 to 0.576) | 0.747 (0.703 to 0.791) | 0.383 (0.345 to 0.421) | 0.550 (0.532 to 0.568) |
| XGBoost | 0.788 (0.776 to 0.799) | 0.565 (0.548 to 0.581) | 0.765 (0.733 to 0.796) | 0.373 (0.349 to 0.397) | 0.547 (0.526 to 0.568) |
| LightGBM | 0.788 (0.776 to 0.799) | 0.566 (0.548 to 0.584) | 0.764 (0.733 to 0.796) | 0.373 (0.338 to 0.408) | 0.548 (0.523 to 0.573) |

**Improvement over linear baseline (lr)** (same folds, difference with 95% interval; interval excluding 0 is consistent across folds):

| Model | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |
|---|---|---|---|---|
| Binned logistic regression (scorecard) | 0.0109 (0.0034 to 0.0184) | 0.0255 (0.0169 to 0.0342) | 0.0015 (-0.0083 to 0.0113) | -0.0168 (-0.0254 to -0.0083) |
| Random forest | 0.0192 (0.0140 to 0.0244) | 0.0393 (0.0267 to 0.0520) | 0.0027 (-0.0528 to 0.0582) | -0.0152 (-0.0242 to -0.0061) |
| XGBoost | 0.0205 (0.0148 to 0.0262) | 0.0428 (0.0303 to 0.0553) | 0.0207 (-0.0396 to 0.0810) | -0.0184 (-0.0287 to -0.0081) |
| LightGBM | 0.0206 (0.0153 to 0.0260) | 0.0442 (0.0324 to 0.0560) | 0.0201 (-0.0259 to 0.0660) | -0.0168 (-0.0281 to -0.0056) |

**Improvement over binned scorecard (lr_binned)**:

| Model | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |
|---|---|---|---|---|
| Random forest | 0.0083 (0.0041 to 0.0124) | 0.0138 (0.0055 to 0.0221) | 0.0012 (-0.0524 to 0.0548) | 0.0017 (-0.0085 to 0.0119) |
| XGBoost | 0.0096 (0.0052 to 0.0140) | 0.0173 (0.0095 to 0.0250) | 0.0192 (-0.0336 to 0.0719) | -0.0016 (-0.0159 to 0.0128) |
| LightGBM | 0.0097 (0.0056 to 0.0138) | 0.0187 (0.0091 to 0.0282) | 0.0186 (-0.0272 to 0.0643) | 0.0000 (-0.0146 to 0.0146) |

**Challenger-vs-challenger paired differences**:

| Comparison | Δ ROC-AUC | Δ PR-AUC | Δ Recall | Δ Cost per customer |
|---|---|---|---|---|
| XGBoost vs Random forest | 0.0013 (-0.0001 to 0.0027) | 0.0035 (0.0002 to 0.0068) | 0.0179 (-0.0401 to 0.0760) | -0.0032 (-0.0086 to 0.0022) |
| LightGBM vs Random forest | 0.0014 (0.0008 to 0.0021) | 0.0049 (0.0013 to 0.0085) | 0.0173 (-0.0319 to 0.0666) | -0.0017 (-0.0101 to 0.0068) |
| LightGBM vs XGBoost | 0.0001 (-0.0008 to 0.0011) | 0.0014 (-0.0020 to 0.0048) | -0.0006 (-0.0517 to 0.0505) | 0.0016 (-0.0036 to 0.0067) |

*Note: Fold-based intervals are indicative and optimistic because cross-validation folds share training data; true out-of-sample uncertainty is somewhat larger. An interval excluding 0 indicates consistent direction across folds.*

## Where we draw the line

A model outputs a probability; we need a cut-off above which a customer is flagged. **Assumption: a missed default costs 5 times as much as a false alarm** (cost 5 vs 1). The cut-off is chosen on training data only to minimise that cost. If the real ratio differs, the cut-off and the 'cost per customer' column change; the ranking of models is the more robust finding.

![Gain and lift](../results/figures/gain_lift.png)

## What drives risk

- **repayment status last month (delay in months)**: higher values raise risk.
- **number of the last 3 months with a payment delay**: higher values raise risk.
- **number of the last 6 months with a payment delay**: higher values raise risk.
- **worst payment delay (months) in the last 6 months**: higher values raise risk.
- **bill amount, month 1 (1 = most recent)**: higher values lower risk.

Sense check: recent late payments are among the top three drivers, as expected.

## Recommendation

**XGBoost**, **Binned logistic regression (scorecard)**, **Random forest** and **LightGBM** all achieve lower cost than the linear scorecard across folds, while mutual cost differences between them include zero. Notably, the binned scorecard matches tree models on cost, eliminating the tree cost advantage. We recommend evaluating either gradient-boosted challenger (for its modest ranking gain) beside the binned scorecard, while keeping the scorecard as the transparent audit reference.

## Limitations

- One dataset from one bank in one country (Taiwan, 2005); results may not transfer.
- Random cross-validation, **no time-based validation**: we have not tested performance on later periods or under economic change.
- **No fairness or regulatory analysis.** Sex, age and marital status are in the data; using them may be restricted in real lending.
- The cost ratio is an assumption; real costs vary by exposure and recovery.
- 'Days past due' is approximated by the dataset's repayment-status codes (months of delay).
