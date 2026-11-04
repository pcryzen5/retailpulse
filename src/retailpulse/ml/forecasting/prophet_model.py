"""Prophet demand forecaster with GB holidays, parallel fitting, and Croston-SBA fallback.

Implements P2-07 & TRD §5.1:
- One model per series with weekly and yearly seasonality
- GB public holidays inclusion
- Parallel fitting across series via joblib
- Automatic fallback to Croston-SBA for ineligible series (< 90 days history or < 20 sales days)
- Lumpy target clipping at 99.5th percentile per series
- Non-negative predictions constraint (yhat >= 0.0)
- Logging of native Prophet intervals for comparison with residual-quantile intervals
- Strict isolation: Prophet is imported only in this module
"""

from __future__ import annotations

import datetime
import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from prophet import Prophet

from retailpulse.features.forecast_features import clip_training_targets
from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.base import Forecaster, resolve_target_dates
from retailpulse.ml.forecasting.baselines import CrostonSBAForecaster

# Suppress verbose Stan and Prophet training logs
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

logger = get_logger(__name__)


def check_series_eligibility(
    series_df: pd.DataFrame,
    *,
    min_history_days: int = 90,
    min_sales_days: int = 20,
) -> tuple[bool, str | None]:
    """Assess if a sales series meets eligibility criteria for Prophet fitting.

    Per TRD §5.1:
    - History span (first sale to latest date) must be >= min_history_days (default 90).
    - Must have >= min_sales_days (default 20) with positive units sold (y > 0).

    Args:
        series_df: Historical daily records for a single series.
        min_history_days: Minimum days span required.
        min_sales_days: Minimum non-zero sales days required.

    Returns:
        (is_eligible, reason_string_if_ineligible).
    """
    if series_df.empty:
        return False, "empty_series"

    positive_sales = series_df[series_df["y"] > 0]
    sales_days_count = len(positive_sales)

    if sales_days_count < min_sales_days:
        return False, "too_few_sales_days"

    first_date = series_df["date"].min()
    last_date = series_df["date"].max()
    history_span = (last_date - first_date).days + 1

    if history_span < min_history_days:
        return False, "history_too_short"

    return True, None


def _fit_single_series(
    sid: Any,
    series_df: pd.DataFrame,
    *,
    country_holidays: str | None,
    weekly_seasonality: bool | str,
    yearly_seasonality: bool | str,
    daily_seasonality: bool | str,
    min_history_days: int,
    min_sales_days: int,
    croston_alpha: float,
    extra_prophet_params: dict[str, Any] | None,
) -> tuple[Any, Any, str, str | None]:
    """Fit either Prophet or Croston-SBA for a single series based on eligibility.

    Top-level helper function to ensure full joblib parallel picklability.
    """
    is_eligible, reason = check_series_eligibility(
        series_df,
        min_history_days=min_history_days,
        min_sales_days=min_sales_days,
    )

    if not is_eligible:
        logger.info(
            "series ineligible for prophet, falling back to croston_sba",
            series_id=sid,
            reason=reason,
        )
        croston = CrostonSBAForecaster(alpha=croston_alpha)
        croston.fit(series_df)
        return sid, croston, "croston_sba", reason

    # Eligible for Prophet
    try:
        prophet_df = pd.DataFrame(
            {
                "ds": pd.to_datetime(series_df["date"]),
                "y": series_df["y"].astype(float),
            }
        )

        params: dict[str, Any] = {
            "weekly_seasonality": weekly_seasonality,
            "yearly_seasonality": yearly_seasonality,
            "daily_seasonality": daily_seasonality,
        }
        if extra_prophet_params:
            params.update(extra_prophet_params)

        model = Prophet(**params)
        if country_holidays:
            model.add_country_holidays(country_name=country_holidays)

        model.fit(prophet_df)
        return sid, model, "prophet", None

    except Exception as exc:
        logger.warning(
            "prophet fitting failed, falling back to croston_sba",
            series_id=sid,
            error=str(exc),
        )
        croston = CrostonSBAForecaster(alpha=croston_alpha)
        croston.fit(series_df)
        return sid, croston, "croston_sba", f"fit_failed: {exc}"


class ProphetForecaster(Forecaster):
    """Local forecasting model fitting Prophet per series with Croston-SBA fallback."""

    name: str = "prophet"

    def __init__(
        self,
        *,
        n_jobs: int = -1,
        country_holidays: str | None = "GB",
        weekly_seasonality: bool | str = True,
        yearly_seasonality: bool | str = "auto",
        daily_seasonality: bool | str = False,
        min_history_days: int = 90,
        min_sales_days: int = 20,
        clip_training_percentile: float | None = 0.995,
        croston_alpha: float = 0.1,
        extra_prophet_params: dict[str, Any] | None = None,
    ) -> None:
        """Initialize Prophet forecaster.

        Args:
            n_jobs: Number of parallel workers for fitting models across series (-1 for all CPUs).
            country_holidays: Country code for holidays integration (defaults to 'GB').
            weekly_seasonality: Fit weekly seasonality component.
            yearly_seasonality: Fit yearly seasonality component ('auto', True, False).
            daily_seasonality: Fit daily sub-day seasonality component.
            min_history_days: Minimum days span required for Prophet eligibility.
            min_sales_days: Minimum non-zero sales days required for Prophet eligibility.
            clip_training_percentile: Percentile for training spike clipping (TRD §5.1).
            croston_alpha: Smoothing parameter for fallback Croston-SBA forecaster.
            extra_prophet_params: Additional kwargs passed to Prophet constructor.
        """
        self.n_jobs = n_jobs
        self.country_holidays = country_holidays
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.min_history_days = min_history_days
        self.min_sales_days = min_sales_days
        self.clip_training_percentile = clip_training_percentile
        self.croston_alpha = croston_alpha
        self.extra_prophet_params = extra_prophet_params or {}

        self._fitted_models: dict[Any, Any] = {}
        self._model_types: dict[Any, str] = {}
        self._ineligibility_reasons: dict[Any, str] = {}

    @property
    def model_types(self) -> dict[Any, str]:
        """Dictionary mapping series_id to the model used ('prophet' or 'croston_sba')."""
        return dict(self._model_types)

    @property
    def ineligibility_reasons(self) -> dict[Any, str]:
        """Dictionary mapping series_id to ineligibility or fallback reason."""
        return dict(self._ineligibility_reasons)

    def fit(self, history: pd.DataFrame) -> ProphetForecaster:
        """Fit individual Prophet models (or Croston-SBA fallbacks) across all series.

        Args:
            history: Historical sales series DataFrame containing:
                series_id, date, y, ...

        Returns:
            Fitted ProphetForecaster instance.
        """
        if history.empty:
            raise ValueError("Cannot fit ProphetForecaster on empty history DataFrame")

        train_hist = history
        if self.clip_training_percentile is not None and "y" in history.columns:
            train_hist = clip_training_targets(
                history,
                target_col="y",
                group_col="series_id",
                quantile=self.clip_training_percentile,
            )

        groups = [(sid, group.copy()) for sid, group in train_hist.groupby("series_id")]

        # Execute fitting sequentially or in parallel
        if self.n_jobs == 1 or len(groups) <= 1:
            results = [
                _fit_single_series(
                    sid,
                    grp,
                    country_holidays=self.country_holidays,
                    weekly_seasonality=self.weekly_seasonality,
                    yearly_seasonality=self.yearly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    min_history_days=self.min_history_days,
                    min_sales_days=self.min_sales_days,
                    croston_alpha=self.croston_alpha,
                    extra_prophet_params=self.extra_prophet_params,
                )
                for sid, grp in groups
            ]
        else:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_series)(
                    sid,
                    grp,
                    country_holidays=self.country_holidays,
                    weekly_seasonality=self.weekly_seasonality,
                    yearly_seasonality=self.yearly_seasonality,
                    daily_seasonality=self.daily_seasonality,
                    min_history_days=self.min_history_days,
                    min_sales_days=self.min_sales_days,
                    croston_alpha=self.croston_alpha,
                    extra_prophet_params=self.extra_prophet_params,
                )
                for sid, grp in groups
            )

        self._fitted_models.clear()
        self._model_types.clear()
        self._ineligibility_reasons.clear()

        for sid, model, m_type, reason in results:
            self._fitted_models[sid] = model
            self._model_types[sid] = m_type
            if reason is not None:
                self._ineligibility_reasons[sid] = reason

        logger.info(
            "fitted prophet forecaster across series",
            total_series=len(groups),
            prophet_count=sum(1 for t in self._model_types.values() if t == "prophet"),
            croston_fallback_count=sum(1 for t in self._model_types.values() if t == "croston_sba"),
        )
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast future demand for all series over the specified horizon.

        Args:
            origin: Last observed date of training/inference cutoff.
            horizon: Number of forecast periods (trading days) ahead.
            calendar_df: Optional dim_date DataFrame for trading-day resolution.

        Returns:
            DataFrame with columns:
                [series_id, date, yhat, prophet_yhat_lower, prophet_yhat_upper, model_type]
            where yhat >= 0.0.
        """
        if not self._fitted_models:
            raise ValueError("ProphetForecaster must be fitted before predict() can be called")

        target_dates = resolve_target_dates(origin, horizon, calendar_df)
        if not target_dates:
            return pd.DataFrame(
                columns=[
                    "series_id",
                    "date",
                    "yhat",
                    "prophet_yhat_lower",
                    "prophet_yhat_upper",
                    "model_type",
                ]
            )

        all_preds: list[pd.DataFrame] = []

        for sid, model in self._fitted_models.items():
            m_type = self._model_types[sid]

            if m_type == "croston_sba":
                croston_preds = model.predict(origin, horizon, calendar_df)
                croston_preds["prophet_yhat_lower"] = croston_preds["yhat"]
                croston_preds["prophet_yhat_upper"] = croston_preds["yhat"]
                croston_preds["model_type"] = "croston_sba"
                all_preds.append(croston_preds)

            else:
                future_df = pd.DataFrame({"ds": pd.to_datetime(target_dates)})
                fcst = model.predict(future_df)

                yhat = np.maximum(0.0, fcst["yhat"].to_numpy(dtype=float))
                lower = np.maximum(0.0, fcst.get("yhat_lower", yhat).to_numpy(dtype=float))
                upper = np.maximum(yhat, fcst.get("yhat_upper", yhat).to_numpy(dtype=float))

                df_out = pd.DataFrame(
                    {
                        "series_id": sid,
                        "date": target_dates,
                        "yhat": yhat,
                        "prophet_yhat_lower": lower,
                        "prophet_yhat_upper": upper,
                        "model_type": "prophet",
                    }
                )
                all_preds.append(df_out)

        combined = pd.concat(all_preds, ignore_index=True)
        return combined.sort_values(["series_id", "date"]).reset_index(drop=True)
