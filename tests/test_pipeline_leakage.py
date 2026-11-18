import importlib.util
import unittest

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from creditrisk.config import FEATURES
from creditrisk.data import prepare_from_raw
from creditrisk.leakage_checks import assert_no_target_or_id, single_feature_auc
from creditrisk.pipelines import make_pipeline, signed_log1p
from creditrisk.synthetic import make_synthetic

HAS_IMB = importlib.util.find_spec("imblearn") is not None


class TestLeakageSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = prepare_from_raw(make_synthetic(1200, seed=5), "synthetic")

    def test_feature_matrix_has_no_target_or_id(self):
        self.assertEqual(list(self.ds.X.columns), FEATURES)
        assert_no_target_or_id(self.ds.X.columns)

    def test_no_single_feature_is_a_label_proxy(self):
        auc = single_feature_auc(self.ds.X, self.ds.y)
        self.assertTrue(((auc < 0.85) & (auc > 0.15)).all(), auc.head())

    def test_scaler_is_fitted_on_training_rows_only(self):
        X, y = self.ds.X, self.ds.y
        tr = np.arange(0, 800)
        pipe = make_pipeline("lr").fit(X.iloc[tr], y.iloc[tr])
        prep = pipe.named_steps["prep"]
        scaler = prep.named_transformers_["amt"].named_steps["scale"]
        train_mean = signed_log1p(X.iloc[tr][prep.transformers_[0][2]].to_numpy()).mean(0)
        full_mean = signed_log1p(X[prep.transformers_[0][2]].to_numpy()).mean(0)
        np.testing.assert_allclose(scaler.mean_, train_mean)
        self.assertFalse(np.allclose(scaler.mean_, full_mean))

    def test_validation_rows_do_not_influence_fit(self):
        """Changing validation features/labels must not change the fitted model."""
        X, y = self.ds.X.copy(), self.ds.y.copy()
        tr = np.arange(0, 800)
        p1 = make_pipeline("lr").fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[800:810])
        X.iloc[800:, :] = X.iloc[800:, :] * 0 + 1e6
        y.iloc[800:] = 1 - y.iloc[800:]
        p2 = make_pipeline("lr").fit(X.iloc[tr], y.iloc[tr]).predict_proba(self.ds.X.iloc[800:810])
        np.testing.assert_allclose(p1, p2)

    def test_cross_val_runs_for_lr_and_rf(self):
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        for m in ("lr", "rf"):
            pipe = make_pipeline(m).set_params(**({"clf__n_estimators": 30} if m == "rf" else {}))
            s = cross_val_score(pipe, self.ds.X, self.ds.y, cv=cv, scoring="roc_auc")
            self.assertTrue((s > 0.55).all(), (m, s))

    @unittest.skipUnless(HAS_IMB, "imbalanced-learn not installed")
    def test_resampling_only_applies_at_fit_time(self):
        X, y = self.ds.X, self.ds.y
        for strat in ("undersample", "smote"):
            pipe = make_pipeline("lr", strat).fit(X.iloc[:800], y.iloc[:800])
            self.assertEqual(len(pipe.predict_proba(X.iloc[800:])), len(X) - 800)   # validation not resampled
            self.assertIn("sampler", pipe.named_steps)

    def test_nan_passthrough_for_xgb_lgbm_none_and_imputation_for_smote(self):
        """xgb and lgbm under 'none' must retain NaNs natively; smote and rf must impute."""
        X_with_nan = self.ds.X.copy()
        X_with_nan.loc[X_with_nan.index[:5], "delay_trend"] = np.nan

        for m in ("xgb", "lgbm"):
            p_none = make_pipeline(m, strategy="none")
            Xt_none = p_none.named_steps["prep"].fit_transform(X_with_nan)
            self.assertTrue(Xt_none.isna().any().any(), f"{m} under 'none' should preserve NaNs natively")

            p_smote = make_pipeline(m, strategy="smote")
            Xt_smote = p_smote.named_steps["prep"].fit_transform(X_with_nan)
            self.assertFalse(Xt_smote.isna().any().any(), f"{m} under 'smote' must impute NaNs for k-NN")

        p_rf = make_pipeline("rf", strategy="none")
        Xt_rf = p_rf.named_steps["prep"].fit_transform(X_with_nan)
        self.assertFalse(Xt_rf.isna().any().any(), "rf under 'none' must impute NaNs")

    def test_lr_binned_pipeline_structure_and_scoring(self):
        """lr_binned must one-hot encode pay_1..pay_6 into 6 levels and run CV safely."""
        pipe = make_pipeline("lr_binned")
        Xt = pipe.named_steps["prep"].fit_transform(self.ds.X)
        cols = list(Xt.columns)
        for k in range(1, 7):
            for lvl in (-2, -1, 0, 1, 2, 3):
                self.assertIn(f"pay_{k}_{lvl}", cols)
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        s = cross_val_score(pipe, self.ds.X, self.ds.y, cv=cv, scoring="roc_auc")
        self.assertTrue((s > 0.55).all(), s)



if __name__ == "__main__":
    unittest.main()
