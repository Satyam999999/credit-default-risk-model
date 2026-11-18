import unittest

import numpy as np
import pandas as pd

from creditrisk.config import ENGINEERED, RAW_COLUMNS
from creditrisk.features import add_engineered_features, masked_slope
from creditrisk.synthetic import make_synthetic
from creditrisk.cleaning import clean_categories, drop_exact_duplicates


def customer_a(id_=1):
    """Hand-built customer. Chronological (oldest -> newest):
    status [-1,0,1,2,0,0]; bill [1000,2000,0,500,500,1000]; payment [0,1000,500,0,250,500]; limit 10000.
    """
    row = {c: 0 for c in RAW_COLUMNS}
    row.update(id=id_, limit_bal=10000.0, sex=1, education=2, marriage=1, age=30)
    status, bill, pay = [-1, 0, 1, 2, 0, 0], [1000, 2000, 0, 500, 500, 1000], [0, 1000, 500, 0, 250, 500]
    for t in range(6):                         # t index 0 = oldest -> UCI column 6
        k = 6 - t
        row[f"pay_{k}"], row[f"bill_amt{k}"], row[f"pay_amt{k}"] = status[t], float(bill[t]), float(pay[t])
    return row


class TestFeatures(unittest.TestCase):
    def test_hand_computed_values(self):
        f = add_engineered_features(pd.DataFrame([customer_a()])).iloc[0]
        # ratios t=2..6: 1000/1000=1, 500/2000=.25, prev bill 0 -> NaN, 250/500=.5, 500/500=1
        self.assertAlmostEqual(f.avg_pay_ratio, 0.6875)
        self.assertAlmostEqual(f.pay_ratio_last3, 0.75)
        self.assertAlmostEqual(f.pay_ratio_trend, 0.025)          # OLS slope over the 4 non-null points
        self.assertEqual(f.n_months_delinquent, 2)
        self.assertEqual(f.n_delinquent_last3, 1)
        self.assertEqual(f.max_delay, 2)
        self.assertAlmostEqual(f.delay_trend, 3 / 105)
        self.assertAlmostEqual(f.avg_utilization, 0.5 / 6)
        self.assertAlmostEqual(f.utilization_latest, 0.1)
        self.assertAlmostEqual(f.utilization_trend, -1.2 / 105)

    def test_negative_bills_and_undefined_ratio(self):
        row = customer_a()
        for k in range(1, 7):
            row[f"bill_amt{k}"] = -10.0
        f = add_engineered_features(pd.DataFrame([row])).iloc[0]
        self.assertTrue(np.isnan(f.avg_pay_ratio) and np.isnan(f.pay_ratio_trend))
        self.assertEqual(f.avg_utilization, 0.0)                  # credit balance -> zero utilization

    def test_ratio_capped_at_one(self):
        row = customer_a()
        row["pay_amt1"] = 1e9
        f = add_engineered_features(pd.DataFrame([row])).iloc[0]
        self.assertLessEqual(f.pay_ratio_last3, 1.0)

    def test_masked_slope_matches_polyfit(self):
        y = np.array([[1.0, 2.0, 4.0, 7.0, 11.0, 16.0], [1, np.nan, 3, np.nan, 5, 6]])
        x = np.arange(1, 7.0)
        s = masked_slope(y, x)
        self.assertAlmostEqual(s[0], np.polyfit(x, y[0], 1)[0])
        ok = ~np.isnan(y[1])
        self.assertAlmostEqual(s[1], np.polyfit(x[ok], y[1][ok], 1)[0])

    def test_all_engineered_columns_present(self):
        out = add_engineered_features(pd.DataFrame([customer_a()]))
        self.assertTrue(set(ENGINEERED) <= set(out.columns))

    def test_features_are_row_local_no_cross_customer_statistics(self):
        """Leakage guard: a customer's features must not change when other customers are removed."""
        df = clean_categories(drop_exact_duplicates(make_synthetic(600, seed=3)))
        full = add_engineered_features(df).set_index("id")[ENGINEERED]
        sub = add_engineered_features(df.sample(100, random_state=0)).set_index("id")[ENGINEERED]
        pd.testing.assert_frame_equal(full.loc[sub.index], sub, check_exact=False, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
