import unittest

import numpy as np

from creditrisk.evaluation import (best_threshold_cost, confusion_counts, fold_ci, gain_lift_table, ks_statistic,
                                   metrics_at_threshold)


class TestEvaluation(unittest.TestCase):
    def test_ks_perfect_and_random(self):
        self.assertAlmostEqual(ks_statistic([0, 0, 1, 1], [.1, .2, .8, .9]), 1.0)
        rng = np.random.default_rng(0)
        self.assertLess(ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)), 0.06)

    def test_confusion_and_metrics(self):
        y, p = [1, 1, 0, 0, 0, 1], [.9, .4, .6, .2, .1, .7]
        m = metrics_at_threshold(y, p, 0.5, c_fn=5, c_fp=1)
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (2, 1, 1, 2))
        self.assertAlmostEqual(m["precision"], 2 / 3)
        self.assertAlmostEqual(m["recall"], 2 / 3)
        self.assertAlmostEqual(m["cost_per_customer"], (5 * 1 + 1 * 1) / 6)

    def test_threshold_moves_down_when_missed_defaults_cost_more(self):
        rng = np.random.default_rng(1)
        y = (rng.random(4000) < 0.25).astype(int)
        p = np.clip(0.25 + 0.25 * (y - 0.25) * 2 + rng.normal(0, .2, 4000), 0.001, 0.999)
        t_equal = best_threshold_cost(y, p, 1, 1)
        t_costly = best_threshold_cost(y, p, 5, 1)
        self.assertLess(t_costly, t_equal)

    def test_threshold_minimizes_cost_vs_grid(self):
        rng = np.random.default_rng(2)
        y = (rng.random(1500) < .2).astype(int)
        p = np.clip(.2 + .3 * (y - .2) + rng.normal(0, .15, 1500), 0, 1)
        t = best_threshold_cost(y, p, 5, 1)
        c = lambda th: metrics_at_threshold(y, p, th, 5, 1)["cost_per_customer"]
        self.assertLessEqual(c(t), min(c(g) for g in np.linspace(0, 1, 201)) + 1e-12)

    def test_flag_nobody_when_scores_uninformative_and_fn_cheap(self):
        y = np.array([0] * 90 + [1] * 10)
        p = np.full(100, 0.5)
        t = best_threshold_cost(y, p, c_fn=1, c_fp=1)
        self.assertEqual(confusion_counts(y, p >= t)["tp"] + confusion_counts(y, p >= t)["fp"], 0)

    def test_gain_lift_table(self):
        y = np.array([1] * 20 + [0] * 80)
        p = np.r_[np.linspace(.9, .5, 20), np.linspace(.4, 0, 80)]
        g = gain_lift_table(y, p, 10)
        self.assertEqual(g.n.sum(), 100)
        self.assertAlmostEqual(g.cum_defaults_captured.iloc[-1], 1.0)
        self.assertAlmostEqual(g.lift.iloc[0], 5.0)               # top decile is all defaults, base rate .2

    def test_fold_ci(self):
        m, lo, hi = fold_ci([0.70, 0.72, 0.71, 0.69, 0.73])
        self.assertAlmostEqual(m, 0.71)
        self.assertLess(lo, m); self.assertGreater(hi, m)


if __name__ == "__main__":
    unittest.main()
