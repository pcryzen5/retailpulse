"""LightGBM forecaster using direct multi-step strategy.

Implements P2-06 & TRD §5.1:
- Direct multi-step strategy: (origin, horizon_day h in 1..28) sampled weekly
- Leakage-safe feature set: lags (1, 7, 14, 28), rolling stats (7, 28), days since last sale,
  price ratio, target calendar attributes, horizon day, product_id, abc_class
- Objectives: 'tweedie' (variance power 1.5) and 'regression'
- Deterministic execution: fixed seed, single-thread consistency
- Objective comparison helper (compare_objectives via backtest)
- Non-negative prediction constraint (yhat >= 0.0)
"""

from __future__ import annotations

import datetime
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from retailpulse.etl.calendar import build_calendar_records
from retailpulse.features.forecast_features import (
    build_direct_features_at_origin,
    build_direct_training_dataset,
    clip_training_targets,
)
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.backtest import run_backtest
from retailpulse.ml.forecasting.base import Forecaster

logger = get_logger(__name__)

# Standard direct strategy features per TRD §5.1
FEATURE_COLUMNS = [
    "horizon_day",
    "target_day_of_week",
    "target_month",
    "target_iso_week",
    "target_is_holiday",
    "target_days_to_christmas",
    "target_in_closure_window",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_28",
    "rolling_std_28",
    "days_since_last_sale",
    "price_ratio",
    "product_id",
    "abc_class",
]

CATEGORICAL_COLUMNS = ["product_id", "abc_class"]


def _ensure_calendar(
    calendar_df: pd.DataFrame | None,
    min_date: datetime.date,
    max_date: datetime.date,
) -> pd.DataFrame:
    """Ensure a valid calendar DataFrame spanning from min_date to max_date."""
    if calendar_df is not None and not calendar_df.empty:
        cal_min = calendar_df["date_key"].min()
        cal_max = calendar_df["date_key"].max()
        if cal_min <= min_date and cal_max >= max_date:
            return calendar_df

    records = build_calendar_records(start_date=min_date, end_date=max_date)
    return pd.DataFrame(records)


class LightGBMForecaster(Forecaster):
    """Global direct multi-step forecaster powered by LightGBM."""

    def __init__(
        self,
        *,
        objective: str = "tweedie",
        horizon: int = 28,
        tweedie_variance_power: float = 1.5,
        n_estimators: int = 100,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        sample_step_days: int = 7,
        min_history_days: int = 35,
        clip_training_percentile: float | None = 0.995,
        random_state: int = 42,
        deterministic: bool = True,
        extra_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize LightGBM forecaster.

        Args:
            objective: Loss objective ('tweedie' or 'regression').
            horizon: Number of trading days ahead to forecast.
            tweedie_variance_power: Parameter p in (1, 2) for Tweedie distribution.
            n_estimators: Number of boosting iterations.
            learning_rate: Boosting learning rate.
            num_leaves: Maximum tree leaves for base learners.
            min_child_samples: Minimum number of data needed in a child (leaf).
            sample_step_days: Interval in days between training origin cutoffs.
            min_history_days: Minimum historical days required before first training origin.
            clip_training_percentile: Optional percentile for clipping lumpy training orders.
            random_state: Seed for reproducibility.
            deterministic: Ensure deterministic tree construction.
            extra_params: Additional LightGBM hyperparameters.
        """
        if objective not in ("tweedie", "regression"):
            raise ValueError(
                f"Unsupported objective '{objective}', expected 'tweedie' or 'regression'"
            )

        self.objective = objective
        self.name = f"lightgbm_{objective}"
        self.horizon = horizon
        self.tweedie_variance_power = tweedie_variance_power
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.sample_step_days = sample_step_days
        self.min_history_days = min_history_days
        self.clip_training_percentile = clip_training_percentile
        self.random_state = random_state
        self.deterministic = deterministic
        self.extra_params = extra_params or {}

        self._model: lgb.LGBMRegressor | None = None
        self._history: pd.DataFrame | None = None
        self._calendar_df: pd.DataFrame | None = None
        self._category_dtypes: dict[str, Any] = {}
        self._fitted_feature_names: list[str] = []

    def fit(
        self,
        history: pd.DataFrame,
        calendar_df: pd.DataFrame | None = None,
    ) -> LightGBMForecaster:
        """Fit LightGBM model across direct multi-step training origins.

        Args:
            history: Historical sales DataFrame with columns:
                series_id, product_id, date, y, price, is_holiday, abc_class.
            calendar_df: Optional dim_date DataFrame. If omitted, built automatically.

        Returns:
            Fitted LightGBMForecaster.
        """
        if history.empty:
            raise ValueError("Cannot fit LightGBMForecaster on empty history DataFrame")

        self._history = history.copy()
        min_date = history["date"].min()
        max_date = history["date"].max() + datetime.timedelta(days=self.horizon + 30)
        self._calendar_df = _ensure_calendar(calendar_df, min_date, max_date)

        # Apply target clipping if enabled
        train_hist = history
        if self.clip_training_percentile is not None and "y" in history.columns:
            train_hist = clip_training_targets(
                history,
                target_col="y",
                group_col="series_id",
                quantile=self.clip_training_percentile,
            )

        # Adapt min_history_days if dataset history is relatively short
        total_days = (history["date"].max() - history["date"].min()).days
        eff_min_history = min(self.min_history_days, max(7, total_days - self.horizon - 7))

        train_dataset = build_direct_training_dataset(
            series_df=train_hist,
            calendar_df=self._calendar_df,
            horizon=self.horizon,
            min_history_days=eff_min_history,
            sample_step_days=self.sample_step_days,
        )

        if train_dataset.empty:
            raise ValueError(
                f"Insufficient historical span to build training dataset (total days: {total_days})"
            )

        features_train = train_dataset[FEATURE_COLUMNS].copy()
        y = train_dataset["y_target"].astype(float).to_numpy()

        self._category_dtypes = {}
        for cat_col in CATEGORICAL_COLUMNS:
            features_train[cat_col] = features_train[cat_col].astype("category")
            self._category_dtypes[cat_col] = features_train[cat_col].cat.categories

        self._fitted_feature_names = list(features_train.columns)

        # Build parameters
        params: dict[str, Any] = {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": min(self.min_child_samples, max(2, len(train_dataset) // 10)),
            "random_state": self.random_state,
            "deterministic": self.deterministic,
            "n_jobs": 1,
            "verbose": -1,
            "importance_type": "gain",
        }

        if self.objective == "tweedie":
            params["tweedie_variance_power"] = self.tweedie_variance_power

        params.update(self.extra_params)

        self._model = lgb.LGBMRegressor(**params)
        self._model.fit(features_train, y)

        logger.info(
            "fitted lightgbm forecaster",
            model_name=self.name,
            objective=self.objective,
            n_training_samples=len(train_dataset),
            n_features=len(self._fitted_feature_names),
        )
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for the specified horizon starting strictly after origin.

        Args:
            origin: Forecast origin date.
            horizon: Number of trading days ahead to forecast.
            calendar_df: Optional dim_date DataFrame.

        Returns:
            DataFrame with columns: [series_id, product_id, date, horizon_day, yhat, abc_class]
            where yhat >= 0.0.
        """
        if self._model is None or self._history is None:
            raise ValueError("LightGBMForecaster must be fitted before predict() can be called")

        cal_df = calendar_df if calendar_df is not None else self._calendar_df
        min_date = origin - datetime.timedelta(days=365)
        max_date = origin + datetime.timedelta(days=horizon + 30)
        cal_df = _ensure_calendar(cal_df, min_date, max_date)

        features_df = build_direct_features_at_origin(
            history_df=self._history,
            calendar_df=cal_df,
            origin=origin,
            horizon=horizon,
        )

        if features_df.empty:
            return pd.DataFrame(
                columns=["series_id", "product_id", "date", "horizon_day", "yhat", "abc_class"]
            )

        features_pred = features_df[FEATURE_COLUMNS].copy()
        for cat_col in CATEGORICAL_COLUMNS:
            if cat_col in self._category_dtypes:
                features_pred[cat_col] = pd.Categorical(
                    features_pred[cat_col],
                    categories=self._category_dtypes[cat_col],
                )
            else:
                features_pred[cat_col] = features_pred[cat_col].astype("category")

        raw_preds = self._model.predict(features_pred)
        # Non-negative predictions constraint per Forecaster contract and DB constraint
        yhat = np.maximum(0.0, raw_preds)

        out = pd.DataFrame(
            {
                "series_id": features_df["series_id"],
                "product_id": features_df["product_id"],
                "date": features_df["target_date"],
                "horizon_day": features_df["horizon_day"],
                "yhat": yhat,
                "abc_class": features_df["abc_class"],
            }
        )
        return out.sort_values(["series_id", "date"]).reset_index(drop=True)

    def get_feature_importances(self) -> pd.DataFrame:
        """Retrieve table of feature importances by gain."""
        if self._model is None:
            return pd.DataFrame(columns=["feature", "importance"])

        importances = self._model.feature_importances_
        return (
            pd.DataFrame(
                {
                    "feature": self._fitted_feature_names,
                    "importance": importances,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


def compare_objectives(
    series_df: pd.DataFrame,
    calendar_df: pd.DataFrame | None = None,
    *,
    horizon: int = 28,
    n_folds: int = 3,
    n_estimators: int = 100,
    random_state: int = 42,
    sample_step_days: int = 7,
    **kwargs: Any,
) -> tuple[LightGBMForecaster, dict[str, Any]]:
    """Compare 'tweedie' vs 'regression' objectives using rolling-origin backtest.

    Per TRD §5.1:
    - Try objectives tweedie and regression; keep the better by backtest.
    - Evaluates both models over rolling-origin folds and compares global WAPE.

    Args:
        series_df: Continuous sales history for products.
        calendar_df: Calendar dimension DataFrame.
        horizon: Forecast horizon days (default 28).
        n_folds: Number of backtest folds to evaluate.
        n_estimators: Number of boosting iterations per model.
        random_state: Fixed random seed for determinism.
        sample_step_days: Step size in days between candidate training origins.
        **kwargs: Additional parameters forwarded to LightGBMForecaster.

    Returns:
        (best_model, report_dict) where best_model is fitted on all series_df
        with the winning objective, and report_dict contains WAPE scores.
    """
    tweedie_model = LightGBMForecaster(
        objective="tweedie",
        horizon=horizon,
        n_estimators=n_estimators,
        random_state=random_state,
        sample_step_days=sample_step_days,
        **kwargs,
    )
    regression_model = LightGBMForecaster(
        objective="regression",
        horizon=horizon,
        n_estimators=n_estimators,
        random_state=random_state,
        sample_step_days=sample_step_days,
        **kwargs,
    )

    tweedie_res = run_backtest(
        model=tweedie_model,
        series_df=series_df,
        calendar_df=calendar_df,
        horizon=horizon,
        n_folds=n_folds,
    )
    tweedie_wape = float(tweedie_res.global_wape)

    regression_res = run_backtest(
        model=regression_model,
        series_df=series_df,
        calendar_df=calendar_df,
        horizon=horizon,
        n_folds=n_folds,
    )
    regression_wape = float(regression_res.global_wape)

    winner = "tweedie" if tweedie_wape <= regression_wape else "regression"
    logger.info(
        "compared lightgbm objectives",
        winner=winner,
        tweedie_wape=tweedie_wape,
        regression_wape=regression_wape,
    )

    best_model = LightGBMForecaster(
        objective=winner,
        horizon=horizon,
        n_estimators=n_estimators,
        random_state=random_state,
        sample_step_days=sample_step_days,
        **kwargs,
    )
    best_model.fit(series_df, calendar_df=calendar_df)

    report = {
        "winner": winner,
        "tweedie_wape": tweedie_wape,
        "regression_wape": regression_wape,
        "n_folds": n_folds,
        "horizon": horizon,
    }
    return best_model, report
