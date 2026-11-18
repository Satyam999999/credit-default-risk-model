"""Integration test: PostgreSQL views must reproduce the pandas feature mirror.
Skipped unless CREDIT_DB_URL points to a reachable PostgreSQL and psycopg2 is installed."""
import importlib.util
import os
import unittest

import numpy as np

from creditrisk.config import ENGINEERED
from creditrisk.data import prepare_from_raw
from creditrisk.synthetic import make_synthetic

URL = os.environ.get("CREDIT_DB_URL")
OK = bool(URL) and importlib.util.find_spec("psycopg2") is not None and importlib.util.find_spec("sqlalchemy") is not None


@unittest.skipUnless(OK, "set CREDIT_DB_URL (and install psycopg2/sqlalchemy) to run")
class TestSqlParity(unittest.TestCase):
    def test_sql_features_match_pandas(self):
        from creditrisk.db import get_engine, load_raw, read_features
        raw = make_synthetic(800, seed=7)
        eng = get_engine(URL)
        load_raw(eng, raw)
        sql = read_features(eng).set_index("id").sort_index()
        pdf = prepare_from_raw(raw).X.sort_index()
        self.assertEqual(list(sql.index), list(pdf.index))
        for c in ENGINEERED:
            np.testing.assert_allclose(sql[c].astype(float), pdf[c].astype(float), rtol=1e-9, atol=1e-9,
                                       equal_nan=True, err_msg=c)


if __name__ == "__main__":
    unittest.main()
