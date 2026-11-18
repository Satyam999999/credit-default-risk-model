"""Leakage-safe model pipelines.

Every data-dependent step (imputation, scaling, one-hot categories, resampling,
class-weight computation) is INSIDE the Pipeline, so cross-validation refits it on
each training fold only. Resamplers use imblearn's Pipeline, which applies them at
fit time only (never when predicting on a validation fold).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import AMOUNT_COLS, CATEGORICAL, OTHER_NUMERIC, SEED

STRATEGIES = ["none", "class_weight", "undersample", "smote"]


def signed_log1p(x):
    """Stateless transform (nothing fitted): tames heavy-tailed amounts, keeps sign."""
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(model: str, strategy: str = "none") -> ColumnTransformer:
    cat = ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)
    if model == "lr":
        amount = Pipeline([("slog", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                           ("scale", StandardScaler())])
        other = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())])
    elif model == "rf" or strategy == "smote":
        amount = "passthrough"
        other = SimpleImputer(strategy="median", add_indicator=True)
    else:  # xgb / lgbm handle NaN natively
        amount, other = "passthrough", "passthrough"
    pre = ColumnTransformer([("amt", amount, AMOUNT_COLS), ("num", other, OTHER_NUMERIC), cat],
                            verbose_feature_names_out=False)
    pre.set_output(transform="pandas")
    return pre


def _balanced_xgb():
    from xgboost import XGBClassifier

    class BalancedXGB(XGBClassifier):
        """scale_pos_weight computed from the TRAINING fold passed to fit (no outside label info)."""
        def fit(self, X, y, **kw):
            y_arr = np.asarray(y)
            pos = float((y_arr == 1).sum())
            self.set_params(scale_pos_weight=(len(y_arr) - pos) / max(pos, 1.0))
            return super().fit(X, y, **kw)
    return BalancedXGB


def make_classifier(model: str, strategy: str = "none", seed: int = SEED):
    weighted = strategy == "class_weight"
    if model == "lr":
        return LogisticRegression(max_iter=3000, class_weight="balanced" if weighted else None, random_state=seed)
    if model == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed,
                                      class_weight="balanced_subsample" if weighted else None)
    if model == "xgb":
        try:
            cls = _balanced_xgb() if weighted else __import__("xgboost").XGBClassifier
        except ImportError as e:
            raise ImportError("xgboost is not installed (see environment.yml)") from e
        return cls(tree_method="hist", eval_metric="logloss", n_jobs=1, random_state=seed)
    if model == "lgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError("lightgbm is not installed (see environment.yml)") from e
        return LGBMClassifier(n_jobs=1, verbose=-1, random_state=seed, importance_type="gain",
                              class_weight="balanced" if weighted else None)
    raise ValueError(f"unknown model {model!r}")


def make_pipeline(model: str, strategy: str = "none", seed: int = SEED):
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    steps = [("prep", build_preprocessor(model, strategy))]
    if strategy in ("undersample", "smote"):
        try:
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as e:
            raise ImportError("imbalanced-learn is required for resampling strategies") from e
        sampler = RandomUnderSampler(random_state=seed) if strategy == "undersample" else SMOTE(random_state=seed)
        steps.append(("sampler", sampler))
        steps.append(("clf", make_classifier(model, strategy, seed)))
        return ImbPipeline(steps)
    steps.append(("clf", make_classifier(model, strategy, seed)))
    return Pipeline(steps)


# Hyperparameter search spaces (RandomizedSearchCV); keys address the 'clf' step.
SEARCH_SPACES = {
    "lr": {"clf__C": loguniform(1e-3, 1e2)},
    "rf": {"clf__n_estimators": randint(200, 500), "clf__max_depth": [6, 8, 12, 16, None],
           "clf__min_samples_leaf": randint(5, 60), "clf__max_features": ["sqrt", 0.3, 0.5]},
    "xgb": {"clf__n_estimators": randint(150, 600), "clf__learning_rate": loguniform(0.01, 0.2),
            "clf__max_depth": randint(3, 8), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 11),
            "clf__reg_lambda": loguniform(0.1, 20)},
    "lgbm": {"clf__n_estimators": randint(150, 600), "clf__learning_rate": loguniform(0.01, 0.2),
             "clf__num_leaves": randint(15, 64), "clf__min_child_samples": randint(20, 120),
             "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1],
             "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(0.1, 20)},
}

# Fixed, sensible parameters for the imbalance study (isolates the effect of the strategy).
DEFAULT_PARAMS = {
    "lr": {"clf__C": 1.0},
    "rf": {"clf__n_estimators": 300, "clf__max_depth": 8, "clf__min_samples_leaf": 20},
    "xgb": {"clf__n_estimators": 300, "clf__learning_rate": 0.05, "clf__max_depth": 4,
            "clf__subsample": 0.8, "clf__colsample_bytree": 0.8},
    "lgbm": {"clf__n_estimators": 300, "clf__learning_rate": 0.05, "clf__num_leaves": 31,
             "clf__min_child_samples": 50, "clf__subsample": 0.8, "clf__subsample_freq": 1,
             "clf__colsample_bytree": 0.8},
}
