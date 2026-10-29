"""Base Forecaster protocol and shared forecasting abstractions.

Implements P2-03 & TRD §5.1:
- Forecaster protocol definition
- Standardized predict return contract: [series_id, date, yhat] with yhat >= 0
- Trading day horizon projection helper
"""

from __future__ import annotations

import datetime
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    """Protocol defining standard lifecycle for retail demand forecasting models."""

    name: str

    def fit(self, history: pd.DataFrame) -> Forecaster:
        """Fit model parameters on historical sales series.

        Args:
            history: DataFrame with columns:
                series_id, date, y, price, is_holiday, abc_class (plus model-specific).

        Returns:
            Fitted Forecaster instance.
        """
        ...

    def predict(
        self,
        origin: datetime.date,
        horizon: int,
        calendar_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate forecasts for the specified horizon starting strictly after origin.

        Args:
            origin: Last observed date of the training/inference cutoff.
            horizon: Number of forecast periods (trading days) ahead.
            calendar_df: Optional dim_date DataFrame for trading-day alignment.

        Returns:
            DataFrame with columns: [series_id, date, yhat] where yhat >= 0.
        """
        ...


def resolve_target_dates(
    origin: datetime.date,
    horizon: int,
    calendar_df: pd.DataFrame | None = None,
) -> list[datetime.date]:
    """Resolve the list of target forecast dates following the origin date.

    If calendar_df is supplied, selects the next `horizon` trading days (is_trading_day = True).
    Otherwise, defaults to the next `horizon` consecutive calendar days.
    """
    if horizon <= 0:
        return []

    if calendar_df is not None and not calendar_df.empty:
        cal_mask = (calendar_df["date_key"] > origin) & (
            calendar_df["is_trading_day"].fillna(True).astype(bool)
        )
        trading_dates = list(
            calendar_df.loc[cal_mask].sort_values("date_key")["date_key"].head(horizon)
        )
        if len(trading_dates) == horizon:
            return trading_dates

    # Fallback to consecutive calendar days
    return [origin + datetime.timedelta(days=i) for i in range(1, horizon + 1)]
