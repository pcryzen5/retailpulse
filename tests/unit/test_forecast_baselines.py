"""Unit tests for Forecaster protocol and statistical baseline models.

Verifies P2-03 & TRD §5.1:
- Forecaster protocol adherence
- NaiveForecaster: repeats last observed value
- SeasonalNaiveForecaster: repeats values with weekly periodicity (m=7)
- MovingAverageForecaster: mean of last 28 trading days
- CrostonSBAForecaster: Croston with Syntetos-Boylan bias correction for intermittent demand
- Multi-series handling and non-negative predictions constraint
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.baselines import (
    CrostonSBAForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def _make_series(
    y_values: list[float],
    series_id: int | str = 1,
    start_date: datetime.date = datetime.date(2011, 1, 1),
) -> pd.DataFrame:
    """Helper to create a standard history DataFrame for testing."""
    dates = [start_date + datetime.timedelta(days=i) for i in range(len(y_values))]
    return pd.DataFrame(
        {
            "series_id": [series_id] * len(y_values),
            "product_id": [series_id] * len(y_values),
            "date": dates,
            "y": y_values,
            "price": [10.0] * len(y_values),
            "is_holiday": [False] * len(y_values),
            "abc_class": ["A"] * len(y_values),
        }
    )


def test_protocol_conformance() -> None:
    """Verify all baseline models conform to the Forecaster Protocol."""
    models: list[Forecaster] = [
        NaiveForecaster(),
        SeasonalNaiveForecaster(),
        MovingAverageForecaster(),
        CrostonSBAForecaster(),
    ]
    for model in models:
        assert isinstance(model.name, str)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")


def test_naive_forecaster_constant_prediction() -> None:
    """Verify NaiveForecaster returns the last observed value for all horizon steps."""
    history = _make_series([10.0, 15.0, 25.0, 42.0])
    origin = datetime.date(2011, 1, 4)

    forecaster = NaiveForecaster()
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=7)

    assert len(preds) == 7
    assert (preds["yhat"] == 42.0).all()
    assert (preds["series_id"] == 1).all()
    assert preds["date"].iloc[0] == origin + datetime.timedelta(days=1)
    assert preds["date"].iloc[-1] == origin + datetime.timedelta(days=7)


def test_naive_forecaster_negative_clipping() -> None:
    """Verify NaiveForecaster clips negative last value to 0.0."""
    history = _make_series([10.0, -5.0])
    origin = datetime.date(2011, 1, 2)

    forecaster = NaiveForecaster()
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=3)

    assert (preds["yhat"] == 0.0).all()


def test_seasonal_naive_forecaster_weekly_pattern() -> None:
    """Verify SeasonalNaiveForecaster replicates the 7-day pattern cyclically."""
    weekly_cycle = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    history = _make_series(weekly_cycle * 3)  # 21 days
    origin = datetime.date(2011, 1, 21)

    forecaster = SeasonalNaiveForecaster(seasonal_period=7)
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=14)

    assert len(preds) == 14
    expected_yhat = weekly_cycle + weekly_cycle
    assert list(preds["yhat"]) == pytest.approx(expected_yhat)


def test_seasonal_naive_fallback_when_short_history() -> None:
    """Verify SeasonalNaive falls back to naive last-value if history < seasonal_period."""
    history = _make_series([10.0, 20.0, 30.0])  # only 3 days (< 7)
    origin = datetime.date(2011, 1, 3)

    forecaster = SeasonalNaiveForecaster(seasonal_period=7)
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=5)

    assert len(preds) == 5
    assert (preds["yhat"] == 30.0).all()


def test_moving_average_forecaster_28_days() -> None:
    """Verify MovingAverageForecaster computes mean of last 28 trading days."""
    # 35 days: first 7 days are 100, last 28 days are 14 days of 10 and 14 days of 20
    # Last 28 days mean: (14*10 + 14*20) / 28 = 15.0
    y_values = [100.0] * 7 + [10.0] * 14 + [20.0] * 14
    history = _make_series(y_values)
    origin = history["date"].iloc[-1]

    forecaster = MovingAverageForecaster(window_size=28)
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=10)

    assert len(preds) == 10
    assert np.allclose(preds["yhat"], 15.0)


def test_moving_average_short_history() -> None:
    """Verify MovingAverageForecaster handles history shorter than 28 days."""
    history = _make_series([10.0, 20.0, 30.0])  # mean = 20.0
    origin = history["date"].iloc[-1]

    forecaster = MovingAverageForecaster(window_size=28)
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=4)

    assert len(preds) == 4
    assert np.allclose(preds["yhat"], 20.0)


def test_croston_sba_forecaster_intermittent_demand() -> None:
    """Verify CrostonSBAForecaster calculates Syntetos-Boylan corrected forecast."""
    # Intermittent series: demand of 10 every 4 days
    # Periods: 10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0, 10
    y_values = [10.0, 0.0, 0.0, 0.0] * 4
    history = _make_series(y_values)
    origin = history["date"].iloc[-1]

    forecaster = CrostonSBAForecaster(alpha=0.1)
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=7)

    assert len(preds) == 7
    pred_val = float(preds["yhat"].iloc[0])
    # Inter-arrival interval is 4.0, size is 10.0 -> z/p = 2.5
    # SBA factor for alpha=0.1 is 1 - 0.05 = 0.95 -> 2.5 * 0.95 = 2.375
    assert pred_val == pytest.approx(2.375)


def test_croston_sba_all_zeros() -> None:
    """Verify CrostonSBAForecaster returns 0.0 when series has zero historical demand."""
    history = _make_series([0.0] * 20)
    origin = datetime.date(2011, 1, 20)

    forecaster = CrostonSBAForecaster()
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=5)

    assert (preds["yhat"] == 0.0).all()


def test_multi_series_forecasting() -> None:
    """Verify all forecasters fit and predict multiple series simultaneously."""
    df1 = _make_series([10.0, 20.0, 30.0], series_id="SKU_A")
    df2 = _make_series([5.0, 5.0, 5.0], series_id="SKU_B")
    combined = pd.concat([df1, df2], ignore_index=True)
    origin = datetime.date(2011, 1, 3)

    for cls in [
        NaiveForecaster,
        SeasonalNaiveForecaster,
        MovingAverageForecaster,
        CrostonSBAForecaster,
    ]:
        model = cls()
        model.fit(combined)
        preds = model.predict(origin=origin, horizon=3)

        assert set(preds["series_id"]) == {"SKU_A", "SKU_B"}
        assert len(preds) == 6  # 2 series * 3 horizon steps
        assert (preds["yhat"] >= 0.0).all()


def test_predict_with_calendar_trading_days() -> None:
    """Verify target dates align with calendar trading days when provided."""
    history = _make_series([10.0, 20.0])
    origin = datetime.date(2011, 1, 2)

    cal_df = pd.DataFrame(
        [
            {"date_key": datetime.date(2011, 1, 3), "is_trading_day": True},
            {"date_key": datetime.date(2011, 1, 4), "is_trading_day": False},
            {"date_key": datetime.date(2011, 1, 5), "is_trading_day": True},
        ]
    )

    forecaster = NaiveForecaster()
    forecaster.fit(history)
    preds = forecaster.predict(origin=origin, horizon=2, calendar_df=cal_df)

    assert len(preds) == 2
    assert list(preds["date"]) == [datetime.date(2011, 1, 3), datetime.date(2011, 1, 5)]
