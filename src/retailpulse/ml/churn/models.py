"""Machine learning models and calibration pipelines for churn prediction.

Implements P3-04, TRD §5.3 & PRD §7.4 (FR-CH-03, FR-CH-04):
- Logistic Regression baseline: median imputation + missing indicator + standard scaling + OHE.
- LightGBM forecaster: native missing value handling, early stopping on validation cutoff.
- Probability calibration: selects between Isotonic Regression and Platt Scaling (Sigmoid)
  based on minimum validation Brier score.
- Unified CalibratedChurnModel container with serialization and inference methods.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from retailpulse.features.churn_features import CHURN_FEATURE_COLUMNS
from retailpulse.logging import get_logger
from retailpulse.ml.churn.metrics import (
    assign_risk_bands,
)

logger = get_logger(__name__)

NUMERIC_FEATURES = [col for col in CHURN_FEATURE_COLUMNS if col != "country"]
CATEGORICAL_FEATURES = ["country"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_logistic_baseline_pipeline(seed: int = 42) -> Pipeline:
    """Create a scikit-learn Pipeline for the Logistic Regression baseline model.

    Features:
      - Numeric: median imputation with missing indicator + StandardScaler
      - Categorical (country): OneHotEncoder (ignoring unseen categories)
      - Estimator: class_weight='balanced' with L2 regularization
    """
    num_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    cat_transformer = Pipeline(
        [
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERIC_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return Pipeline(
        [
            ("prep", preprocessor),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=seed,
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def fit_lightgbm_with_early_stopping(
    x_train: pd.DataFrame,
    y_train: npt.NDArray[Any] | pd.Series,
    x_val: pd.DataFrame,
    y_val: npt.NDArray[Any] | pd.Series,
    *,
    seed: int = 42,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 5,
    num_leaves: int = 31,
    early_stopping_rounds: int = 20,
) -> Pipeline:
    """Fit a LightGBM classifier with validation-set early stopping.

    Args:
        x_train: Training features DataFrame containing ALL_FEATURES.
        y_train: Training labels (0/1).
        x_val: Validation features DataFrame.
        y_val: Validation labels (0/1).
        seed: Random state seed.
        n_estimators: Maximum tree boosting rounds.
        learning_rate: Shrinkage rate.
        max_depth: Maximum tree depth.
        num_leaves: Maximum tree leaves.
        early_stopping_rounds: Rounds without validation improvement before stopping.

    Returns:
        Fitted scikit-learn Pipeline wrapping ColumnTransformer and LGBMClassifier.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    x_train_prep = preprocessor.fit_transform(x_train[ALL_FEATURES])
    x_val_prep = preprocessor.transform(x_val[ALL_FEATURES])

    y_tr_arr = np.asarray(y_train, dtype=np.int64)
    y_val_arr = np.asarray(y_val, dtype=np.int64)

    clf = LGBMClassifier(
        random_state=seed,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        verbosity=-1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(
            x_train_prep,
            y_tr_arr,
            eval_set=[(x_val_prep, y_val_arr)],
            callbacks=[early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)],
        )

    return Pipeline([("prep", preprocessor), ("clf", clf)])


def calibrate_model(
    base_pipeline: Pipeline,
    x_val: pd.DataFrame,
    y_val: npt.NDArray[Any] | pd.Series,
) -> tuple[Any, str | None, float, float]:
    """Calibrate a fitted classifier on validation data, selecting between Isotonic and Sigmoid.

    TRD §5.3: 'Calibrate on validation (isotonic or Platt, whichever has the lower Brier).'

    Args:
        base_pipeline: Pre-fitted classifier or Pipeline.
        x_val: Validation features DataFrame.
        y_val: Validation labels.

    Returns:
        Tuple of (chosen_calibrator, method_name, brier_isotonic, brier_sigmoid).
    """
    y_val_arr = np.asarray(y_val, dtype=np.int64)
    x_val_subset = x_val[ALL_FEATURES]

    if len(np.unique(y_val_arr)) < 2:
        logger.warning(
            "validation set contains only one class; skipping probability calibration",
            classes=np.unique(y_val_arr).tolist(),
        )
        return base_pipeline, None, 0.0, 0.0

    frozen = FrozenEstimator(base_pipeline)

    cal_iso = CalibratedClassifierCV(frozen, method="isotonic").fit(x_val_subset, y_val_arr)
    cal_sig = CalibratedClassifierCV(frozen, method="sigmoid").fit(x_val_subset, y_val_arr)

    p_iso = cal_iso.predict_proba(x_val_subset)[:, 1]
    p_sig = cal_sig.predict_proba(x_val_subset)[:, 1]

    brier_iso = float(brier_score_loss(y_val_arr, p_iso))
    brier_sig = float(brier_score_loss(y_val_arr, p_sig))

    if brier_iso < brier_sig:
        chosen_calibrator = cal_iso
        chosen_method = "isotonic"
    else:
        chosen_calibrator = cal_sig
        chosen_method = "sigmoid"

    logger.info(
        "model calibration evaluated",
        method=chosen_method,
        brier_isotonic=round(brier_iso, 5),
        brier_sigmoid=round(brier_sig, 5),
    )
    return chosen_calibrator, chosen_method, brier_iso, brier_sig


class CalibratedChurnModel:
    """Production wrapper for churn classifier inference and artifact persistence."""

    def __init__(
        self,
        estimator: Any,
        *,
        algorithm: str,
        calibration_method: str | None = None,
        feature_names: list[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.algorithm = algorithm
        self.calibration_method = calibration_method
        self.feature_names = feature_names or list(ALL_FEATURES)

    def predict_proba(self, x: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Predict 1D array of churn probabilities (class 1) in [0, 1]."""
        x_subset = x[self.feature_names]
        probs_2d = self.estimator.predict_proba(x_subset)
        return np.asarray(probs_2d[:, 1], dtype=np.float64)

    def predict(self, x: pd.DataFrame, threshold: float = 0.50) -> npt.NDArray[np.int64]:
        """Predict binary churn labels (0 or 1) using a decision threshold."""
        probs = self.predict_proba(x)
        return (probs >= threshold).astype(np.int64)

    def predict_bands(
        self,
        x: pd.DataFrame,
        *,
        medium_threshold: float = 0.30,
        high_threshold: float = 0.60,
    ) -> npt.NDArray[Any]:
        """Predict canonical risk bands ('low', 'medium', 'high')."""
        probs = self.predict_proba(x)
        return assign_risk_bands(
            probs,
            medium_threshold=medium_threshold,
            high_threshold=high_threshold,
        )

    def save(self, file_path: Path | str) -> None:
        """Serialize model container to disk using joblib."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, file_path: Path | str) -> CalibratedChurnModel:
        """Load serialized model container from disk."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found at {path}")
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: got {type(model)}")
        return model
