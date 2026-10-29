"""Statistical baseline forecasting models.

Implements P2-03 & TRD §5.1:
- NaiveForecaster: Last observed value
- SeasonalNaiveForecaster: Value 7 days earlier (weekly periodicity)
- MovingAverageForecaster: Mean of last 28 trading days
- CrostonSBAForecaster: Croston with Syntetos-Boylan bias correction for intermittent demand
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd

from retailpulse.logging import get_logger
from retailpulse.ml.forecasting.base import Forecaster, resolve_target_dates

logger = get_logger(__name__)


class NaiveForecaster:
    """Naive baseline model forecasting the last observed demand value."""

    name: str = "naive"

    def __init__(self) -> None:
        self._last_values: dict[Any, float] = {}

    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit by storing the most recent observed demand per series."""
        self._last_values.clear()
        for sid, group in history.groupby("series_id"):
            sorted_g = group.sort_values("date")
            if not sorted_g.empty:
                last_y = float(sorted_g["y"].iloc[-1])
                self._last_values[sid] = max(0.0, last_y)
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast the last observed value across all horizon days."""
        target_dates = resolve_target_dates(origin, horizon, calendar_df)
        rows: list[dict[str, Any]] = []

        for sid, last_val in self._last_values.items():
            for d in target_dates:
                rows.append({"series_id": sid, "date": d, "yhat": last_val})

        return pd.DataFrame(rows, columns=["series_id", "date", "yhat"])


class SeasonalNaiveForecaster:
    """Seasonal naive model projecting the demand from m trading days prior (default m=7)."""

    name: str = "seasonal_naive"

    def __init__(self, seasonal_period: int = 7) -> None:
        self.seasonal_period = seasonal_period
        self._seasonal_patterns: dict[Any, list[float]] = {}
        self._fallback_values: dict[Any, float] = {}

    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit by extracting the last m observations per series."""
        self._seasonal_patterns.clear()
        self._fallback_values.clear()

        for sid, group in history.groupby("series_id"):
            sorted_g = group.sort_values("date")
            y_vals = [max(0.0, float(v)) for v in sorted_g["y"]]
            if not y_vals:
                continue

            if len(y_vals) >= self.seasonal_period:
                self._seasonal_patterns[sid] = y_vals[-self.seasonal_period :]
            else:
                self._fallback_values[sid] = y_vals[-1]

        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast repeating the trailing seasonal cycle."""
        target_dates = resolve_target_dates(origin, horizon, calendar_df)
        rows: list[dict[str, Any]] = []

        for sid, pattern in self._seasonal_patterns.items():
            for h, d in enumerate(target_dates):
                val = pattern[h % self.seasonal_period]
                rows.append({"series_id": sid, "date": d, "yhat": val})

        for sid, fallback in self._fallback_values.items():
            for d in target_dates:
                rows.append({"series_id": sid, "date": d, "yhat": fallback})

        return pd.DataFrame(rows, columns=["series_id", "date", "yhat"])


class MovingAverageForecaster:
    """Moving average baseline forecasting the mean of the trailing N days (default 28)."""

    name: str = "moving_average"

    def __init__(self, window_size: int = 28) -> None:
        self.window_size = window_size
        self._means: dict[Any, float] = {}

    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit by computing the mean of the trailing window per series."""
        self._means.clear()
        for sid, group in history.groupby("series_id"):
            sorted_g = group.sort_values("date")
            y_vals = sorted_g["y"].iloc[-self.window_size :].astype(float)
            mean_val = float(y_vals.mean()) if not y_vals.empty else 0.0
            self._means[sid] = max(0.0, mean_val)
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast the historical moving average as a constant level."""
        target_dates = resolve_target_dates(origin, horizon, calendar_df)
        rows: list[dict[str, Any]] = []

        for sid, mean_val in self._means.items():
            for d in target_dates:
                rows.append({"series_id": sid, "date": d, "yhat": mean_val})

        return pd.DataFrame(rows, columns=["series_id", "date", "yhat"])


def _compute_croston_sba_level(y: np.ndarray[Any, Any], alpha: float) -> float:
    """Calculate Croston SBA demand rate for a single series array."""
    non_zero_indices = np.where(y > 0)[0]
    n_nonzero = len(non_zero_indices)
    if n_nonzero == 0:
        return 0.0

    if n_nonzero == 1:
        z = float(y[non_zero_indices[0]])
        p = float(len(y))
        sba_factor = 1.0 - (alpha / 2.0)
        return max(0.0, sba_factor * (z / p))

    # Multiple non-zero demands: compute inter-arrival intervals
    sizes = y[non_zero_indices].astype(float)
    intervals = np.diff(non_zero_indices).astype(float)

    # Standard initialization: mean of observed non-zero sizes and intervals
    z = float(np.mean(sizes))
    p = float(np.mean(intervals))

    for i in range(len(intervals)):
        z = alpha * sizes[i + 1] + (1.0 - alpha) * z
        p = alpha * intervals[i] + (1.0 - alpha) * p

    if p <= 0.0:
        return 0.0

    sba_factor = 1.0 - (alpha / 2.0)
    demand_rate = sba_factor * (z / p)
    return max(0.0, float(demand_rate))


class CrostonSBAForecaster:
    """Croston model with Syntetos-Boylan bias correction (SBA) for intermittent demand.

    Decomposes intermittent series into non-zero demand size and inter-demand intervals,
    applying exponential smoothing and multiplying by (1 - alpha / 2) to eliminate
    positive bias.
    """

    name: str = "croston_sba"

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._rates: dict[Any, float] = {}

    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit Croston-SBA parameters across historical sales series."""
        self._rates.clear()
        for sid, group in history.groupby("series_id"):
            sorted_g = group.sort_values("date")
            y_arr = sorted_g["y"].to_numpy(dtype=float)
            rate = _compute_croston_sba_level(y_arr, self.alpha)
            self._rates[sid] = rate
        return self

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Forecast the constant intermittent demand rate across all horizon days."""
        target_dates = resolve_target_dates(origin, horizon, calendar_df)
        rows: list[dict[str, Any]] = []

        for sid, rate in self._rates.items():
            for d in target_dates:
                rows.append({"series_id": sid, "date": d, "yhat": rate})

        return pd.DataFrame(rows, columns=["series_id", "date", "yhat"])
