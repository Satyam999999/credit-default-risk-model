import unittest

import numpy as np
import pandas as pd

from creditrisk import extra_analyses as E
from creditrisk.data import prepare_from_raw
from creditrisk.run_experiments import nested_cv
from creditrisk.synthetic import make_synthetic


class TestRanking(unittest.TestCase):
    def test_top_share_rounds_to_nearest_customer(self):
        self.assertEqual([E.n_top(5993, q) for q in (0.1, 0.2, 0.3)], [599, 1199, 1798])

    def test_defaults_in_top(self):
        y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        p = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
        self.assertEqual(E.defaults_in_top(y, p, 0.3), (2, 3))

    def test_operational_ranking_paired_difference(self):
        rng = np.random.default_rng(0)
        rows = []
        for k in range(4):                           # 4 folds x 100 customers
            y = (rng.random(100) < .3).astype(int)
            good = y + rng.normal(0, .5, 100)        # informative score
            noise = rng.random(100)                  # uninformative score
            rows.append(pd.DataFrame({"y_true": y, "p_base": noise, "p_chal": good, "fold": k}))
        folds, summ = E.operational_ranking(pd.concat(rows), "base", "chal", quantiles=(0.2,))
        self.assertEqual(len(folds), 4)
        self.assertGreater(summ.diff_defaults.iloc[0], 0)
        self.assertGreater(summ.diff_pp_ci_low.iloc[0], 0)          # informative score clearly wins
        np.testing.assert_allclose(folds.diff_pp, 100 * folds.diff_defaults / folds.defaults)


class TestSummaries(unittest.TestCase):
    def test_grouped_shap_handles_missing_indicators(self):
        df = pd.DataFrame({"feature": ["pay_1", "max_delay", "limit_bal", "delay_trend_missingindicator", "age"],
                           "mean_abs_shap": [0.4, 0.2, 0.1, 0.1, 0.2]})
        g = E.grouped_shap(df)
        self.assertEqual(g["delinquency"]["n_features"], 3)
        self.assertAlmostEqual(g["delinquency"]["share"], 0.7 / 1.0)
        self.assertAlmostEqual(g["delinquency"]["share"] + g["other"]["share"], 1.0)
        self.assertEqual(g["top_feature"], "pay_1")

    def test_cost_summary_baseline_difference_is_zero(self):
        rows = [{"model": m, "fold": k, "fn_cost": 5.0, "cost_per_customer": c + 0.01 * k}
                for m, c in (("lr_binned", .55), ("rf", .54)) for k in range(5)]
        s = E.summarize_cost_sensitivity(pd.DataFrame(rows)).set_index("model")
        self.assertAlmostEqual(s.loc["lr_binned", "diff_vs_lr_binned"], 0.0)
        self.assertAlmostEqual(s.loc["rf", "diff_vs_lr_binned"], -0.01)


class TestCostRefit(unittest.TestCase):
    def test_refit_reproduces_nested_cv_thresholds_and_costs(self):
        """At the main-run ratio the cost-sensitivity refit must equal nested_cv's own fold costs."""
        ds = prepare_from_raw(make_synthetic(900, seed=11), "synthetic")
        X, y = ds.X, ds.y
        oof = pd.DataFrame({"y_true": y.to_numpy()}, index=X.index)
        all_rows, all_params = [], []
        for m in ("lr", "lr_binned", "rf"):
            rows, params, p, thr, fid = nested_cv(X, y, m, "none", 3, 2, 2, "roc_auc", 5.0, 1.0, 42, 1)
            all_rows += rows; all_params += params
            oof[f"p_{m}"] = p
            oof["fold"] = fid
        cfg = {"seed": 42, "inner": 2, "strategy": "none"}
        folds = E.refit_fold_costs(X, y.to_numpy(), oof, all_params, cfg, ratios=(2.0, 5.0, 10.0))
        diff, ok = E.check_against_main_run(folds, pd.DataFrame(all_rows), 5.0)
        self.assertTrue(ok, diff)
        # a costlier missed default must not raise the chosen threshold
        f = folds[(folds.model == "lr") & (folds.fold == 0)].set_index("fn_cost")
        self.assertLessEqual(f.loc[10.0, "threshold"], f.loc[2.0, "threshold"])


if __name__ == "__main__":
    unittest.main()
