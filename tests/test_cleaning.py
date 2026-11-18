import unittest

import pandas as pd

from creditrisk.cleaning import (clean_categories, data_quality_report, drop_exact_duplicates, standardize_columns)
from creditrisk.config import RAW_COLUMNS
from creditrisk.synthetic import make_synthetic


def _row(**kw):
    base = {c: 0 for c in RAW_COLUMNS}
    base.update(id=1, limit_bal=10000, sex=1, education=2, marriage=1, age=30)
    base.update(kw)
    return base


class TestCleaning(unittest.TestCase):
    def test_standardize_renames_uci_columns(self):
        raw = pd.DataFrame([_row()]).rename(columns={"pay_1": "PAY_0", "default_next_month": "default payment next month",
                                                     "id": "ID", "limit_bal": "LIMIT_BAL"})
        out = standardize_columns(raw)
        self.assertEqual(list(out.columns), RAW_COLUMNS)

    def test_standardize_missing_column_raises(self):
        with self.assertRaises(ValueError):
            standardize_columns(pd.DataFrame({"id": [1]}))

    def test_clean_categories_maps_invalid_codes(self):
        df = pd.DataFrame([_row(id=i, education=e, marriage=m) for i, (e, m) in
                           enumerate([(0, 0), (5, 1), (6, 2), (1, 3), (4, 2)], start=1)])
        out = clean_categories(df)
        self.assertEqual(out.education.tolist(), [4, 4, 4, 1, 4])
        self.assertEqual(out.marriage.tolist(), [3, 1, 2, 3, 2])

    def test_drop_exact_duplicates_ignores_id_and_keeps_lowest(self):
        df = pd.DataFrame([_row(id=3), _row(id=1), _row(id=2, age=31)])
        out = drop_exact_duplicates(df)
        self.assertEqual(sorted(out.id), [1, 2])

    def test_dq_report_counts(self):
        df = pd.DataFrame([
            _row(id=1, education=0),                                   # invalid education
            _row(id=2, marriage=0, pay_1=-2),                          # invalid marriage, -2 status
            _row(id=3, bill_amt1=-50),                                 # negative bill
            _row(id=4, pay_1=3, pay_2=0, bill_amt2=100),               # jump 0 -> 3 and delay with a bill
            _row(id=5, pay_amt1=50000),                                # payment above limit
        ])
        rep = data_quality_report(df).set_index("check")["n_rows"]
        self.assertEqual(rep["education code outside documented 1-4"], 1)
        self.assertEqual(rep["marriage code outside documented 1-3"], 1)
        self.assertEqual(rep["rows with any pay status of -2 (undocumented)"], 1)
        self.assertEqual(rep["rows with a negative bill amount"], 1)
        self.assertEqual(rep["rows where delay jumps from <=0 to >=2 in one month"], 1)
        self.assertEqual(rep["rows with a payment above the credit limit"], 1)

    def test_synthetic_has_expected_quirks(self):
        raw = make_synthetic(1500, seed=1)
        rep = data_quality_report(raw).set_index("check")["n_rows"]
        self.assertGreater(rep["education code outside documented 1-4"], 0)
        self.assertGreaterEqual(rep["exact duplicate rows (ignoring id, excluding first copy)"], 10)


if __name__ == "__main__":
    unittest.main()
