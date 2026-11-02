"""Unit tests for rolling-origin backtest harness.

Verifies P2-04 & TRD §5.1:
- Rolling-origin cutoff computation with step = horizon
- Strict assertion preventing data leakage (training data date <= origin)
- Model-agnostic backtesting across multiple folds
- VERIFY REQUIREMENT: Seasonal naive is 100% perfect on a weekly-periodic series
- Multi-series evaluation and scoped metric aggregation
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from retailpulse.ml.forecasting.backtest import (
    BacktestResult,
    assert_no_leakage,
    compute_backtest_origins,
    run_backtest,
)
from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.baselines import (
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
)


def _make_periodic_series(
    pattern: list[float],
    n_cycles: int,
    start_date: datetime.date = datetime.date(2011, 1, 1),
    series_id: int | str = 1,
) -> pd.DataFrame:
    """Generate a repetitive cyclic sales series."""
    y_values = pattern * n_cycles
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


def test_compute_backtest_origins_spacing() -> None:
    """Verify origin generation calculates contiguous non-overlapping test windows."""
    # 200 consecutive dates
    base_date = datetime.date(2011, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(200)]

    origins = compute_backtest_origins(
        dates,
        n_folds=5,
        horizon=28,
        step_days=28,
        min_train_days=30,
    )

    assert len(origins) == 5
    # Chronologically sorted
    assert origins == sorted(origins)

    # Origin differences should be exactly 28 days
    diffs = [(origins[i + 1] - origins[i]).days for i in range(len(origins) - 1)]
    assert all(diff == 28 for diff in diffs)

    # The final origin should have exactly 28 days following it in dates
    last_origin = origins[-1]
    days_after = [d for d in dates if d > last_origin]
    assert len(days_after) == 28


def test_compute_backtest_origins_insufficient_history() -> None:
    """Verify ValueError is raised when dates cannot support requested folds."""
    dates = [datetime.date(2011, 1, 1) + datetime.timedelta(days=i) for i in range(50)]
    with pytest.raises(ValueError, match="Insufficient dates"):
        compute_backtest_origins(dates, n_folds=5, horizon=28, min_train_days=28)


def test_backtest_leakage_assertion() -> None:
    """Verify backtest strictly detects and asserts against training data leakage."""
    origin = datetime.date(2011, 4, 1)
    # Valid training data (all dates <= origin)
    valid_df = pd.DataFrame(
        {
            "series_id": [1, 1],
            "date": [datetime.date(2011, 3, 31), origin],
            "y": [10.0, 15.0],
        }
    )
    # Should not raise
    assert_no_leakage(valid_df, origin, fold_idx=0)

    # Invalid training data (contains date > origin)
    leaky_df = pd.DataFrame(
        {
            "series_id": [1, 1],
            "date": [datetime.date(2011, 3, 31), datetime.date(2011, 4, 5)],  # 4/5 > origin
            "y": [10.0, 15.0],
        }
    )

    with pytest.raises(AssertionError, match="Data leakage detected"):
        assert_no_leakage(leaky_df, origin, fold_idx=1)


def test_backtest_fits_only_historical_data() -> None:
    """Verify Forecaster.fit receives strictly historical observations <= origin."""
    series_df = _make_periodic_series([10.0] * 7, n_cycles=26)

    class LeakageCheckingForecaster(NaiveForecaster):
        def __init__(self) -> None:
            super().__init__()
            self.observed_origins: list[datetime.date] = []

        def fit(self, history: pd.DataFrame) -> Forecaster:
            # The last date in history should be exactly the origin for each fold
            max_d = history["date"].max()
            self.observed_origins.append(max_d)
            return super().fit(history)

    model = LeakageCheckingForecaster()
    result = run_backtest(
        model=model,
        series_df=series_df,
        n_folds=3,
        horizon=14,
        min_train_days=28,
    )

    # Model.fit should have been called 3 times, each with max date == origin
    assert model.observed_origins == result.origins


def test_seasonal_naive_perfect_on_periodic_series() -> None:
    """VERIFY REQUIREMENT: Seasonal naive is 100% perfect on weekly-periodic series.

    A repeating 7-day pattern [1, 2, 3, 4, 5, 6, 7] across 5 folds with horizon=28
    must yield exact zero errors:
    - WAPE = 0.0
    - Bias = 0.0
    - MAE = 0.0
    - RMSE = 0.0
    - All residuals == 0.0
    """
    weekly_pattern = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # 28 cycles = 196 trading days: supports 5 folds x 28 days = 140 test days + 56 train days
    series_df = _make_periodic_series(weekly_pattern, n_cycles=28)

    model = SeasonalNaiveForecaster(seasonal_period=7)
    result: BacktestResult = run_backtest(
        model=model,
        series_df=series_df,
        n_folds=5,
        horizon=28,
        min_train_days=28,
        clip_targets=False,
    )

    assert result.n_folds == 5
    assert result.horizon == 28
    assert len(result.origins) == 5

    # Total predictions = 5 folds x 28 horizon steps = 140
    assert len(result.predictions) == 140

    # Predictions must match actuals exactly
    assert (result.predictions["yhat"] == result.predictions["y"]).all()
    assert (result.predictions["residual"] == 0.0).all()

    # Headline global metrics must be exactly zero
    assert result.global_wape == pytest.approx(0.0)
    assert result.global_bias == pytest.approx(0.0)
    assert result.global_mae == pytest.approx(0.0)
    assert result.global_rmse == pytest.approx(0.0)
    assert result.global_smape == pytest.approx(0.0)


def test_backtest_multi_series_and_scopes() -> None:
    """Verify backtest executes across multiple SKUs and records all metric scopes."""
    s1 = _make_periodic_series([10.0] * 7, n_cycles=26, series_id="SKU_1")
    s2 = _make_periodic_series([20.0] * 7, n_cycles=26, series_id="SKU_2")
    s2["abc_class"] = "B"
    combined = pd.concat([s1, s2], ignore_index=True)

    result = run_backtest(
        model=NaiveForecaster(),
        series_df=combined,
        n_folds=3,
        horizon=14,
        min_train_days=28,
    )

    # 3 folds * 14 horizon steps * 2 series = 84 rows
    assert len(result.predictions) == 84
    assert set(result.predictions["series_id"]) == {"SKU_1", "SKU_2"}

    # Metrics DataFrame must include all 4 scopes
    scopes = set(result.metrics["scope"])
    assert "global" in scopes
    assert "abc" in scopes
    assert "horizon" in scopes
    assert "sku" in scopes


def test_backtest_with_model_factory() -> None:
    """Verify backtest accepts factory callable to re-instantiate fresh models."""
    series_df = _make_periodic_series([5.0] * 7, n_cycles=25)

    result = run_backtest(
        model=lambda: MovingAverageForecaster(window_size=14),
        series_df=series_df,
        n_folds=2,
        horizon=14,
        min_train_days=28,
    )

    assert result.model_name == "moving_average"
    assert result.n_folds == 2
    assert result.global_wape == pytest.approx(0.0)


def test_backtest_empty_dataframe_raises() -> None:
    """Verify empty DataFrame raises ValueError."""
    with pytest.raises(ValueError, match="Cannot run backtest on empty series_df"):
        run_backtest(NaiveForecaster(), pd.DataFrame())
