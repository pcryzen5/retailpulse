"""Unit tests for LightGBM direct multi-step forecaster.

Verifies P2-06 & TRD §5.1:
- Conformance to Forecaster protocol
- Direct multi-step strategy with weekly sampled training cutoffs
- Support for 'tweedie' and 'regression' objectives
- Deterministic behavior: same seed gives identical output
- Objective comparison helper (compare_objectives via backtest)
- VERIFY REQUIREMENT: Beats naive forecaster on a synthetic trending-seasonal dataset
- Verification of non-negative predictions constraint (yhat >= 0)
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.baselines import NaiveForecaster
from retailpulse.ml.forecasting.lgbm_model import LightGBMForecaster, compare_objectives
from retailpulse.ml.forecasting.metrics import wape


def _generate_synthetic_trending_seasonal_data(
    n_days: int = 150,
    n_skus: int = 3,
    start_date: datetime.date = datetime.date(2011, 1, 1),
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic retail sales series with upward trend, weekly seasonality, and noise.

    Returns:
        (series_df, calendar_df)
    """
    rng = np.random.default_rng(random_seed)
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]

    # Weekly pattern (Mon=10 to Sun=70, weekend peak)
    weekly_multipliers = [10.0, 15.0, 20.0, 25.0, 40.0, 60.0, 70.0]

    series_rows: list[dict[str, object]] = []
    for sku_idx in range(1, n_skus + 1):
        sku_scale = 1.0 + (sku_idx - 1) * 0.5
        abc = "A" if sku_idx == 1 else ("B" if sku_idx == 2 else "C")
        price = 15.0 + sku_idx * 5.0

        for t, d in enumerate(dates):
            dow = d.isocalendar()[2] - 1  # 0 to 6
            trend = 0.2 * t  # upward linear trend
            seasonality = weekly_multipliers[dow] * sku_scale
            noise = rng.normal(0.0, 2.0)
            demand = max(0.0, trend + seasonality + noise)

            series_rows.append(
                {
                    "series_id": sku_idx,
                    "product_id": sku_idx,
                    "date": d,
                    "y": demand,
                    "price": price,
                    "is_holiday": False,
                    "abc_class": abc,
                }
            )

    cal_rows = [
        {
            "date_key": d,
            "is_trading_day": True,
            "is_holiday": False,
            "day_of_week": d.isocalendar()[2],
            "month": d.month,
            "iso_week": d.isocalendar()[1],
        }
        for d in dates
    ]

    return pd.DataFrame(series_rows), pd.DataFrame(cal_rows)


def test_protocol_conformance() -> None:
    """Verify LightGBMForecaster conforms to the Forecaster Protocol."""
    model = LightGBMForecaster(objective="tweedie", random_state=42)
    assert isinstance(model, Forecaster)
    assert model.name.startswith("lightgbm")
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_determinism_same_seed_gives_identical_output() -> None:
    """VERIFY REQUIREMENT: Two models fitted with identical seed produce exact same predictions."""
    series_df, cal_df = _generate_synthetic_trending_seasonal_data(n_days=100, n_skus=2)
    cutoff_origin = datetime.date(2011, 3, 20)
    horizon = 14

    train_data = series_df[series_df["date"] <= cutoff_origin].copy()

    model1 = LightGBMForecaster(
        objective="regression",
        n_estimators=30,
        random_state=42,
        horizon=horizon,
    )
    model1.fit(train_data, calendar_df=cal_df)
    preds1 = model1.predict(origin=cutoff_origin, horizon=horizon, calendar_df=cal_df)

    model2 = LightGBMForecaster(
        objective="regression",
        n_estimators=30,
        random_state=42,
        horizon=horizon,
    )
    model2.fit(train_data, calendar_df=cal_df)
    preds2 = model2.predict(origin=cutoff_origin, horizon=horizon, calendar_df=cal_df)

    assert not preds1.empty
    assert not preds2.empty
    assert len(preds1) == len(preds2)
    np.testing.assert_allclose(preds1["yhat"].to_numpy(), preds2["yhat"].to_numpy(), rtol=1e-5)


def test_beats_naive_on_trending_seasonal_dataset() -> None:
    """VERIFY REQUIREMENT: LightGBM beats naive baseline on synthetic trending-seasonal series."""
    series_df, cal_df = _generate_synthetic_trending_seasonal_data(n_days=140, n_skus=2)
    cutoff_origin = datetime.date(2011, 4, 15)
    horizon = 28

    train_data = series_df[series_df["date"] <= cutoff_origin].copy()
    test_actuals = series_df[
        (series_df["date"] > cutoff_origin)
        & (series_df["date"] <= cutoff_origin + datetime.timedelta(days=horizon))
    ].copy()

    # 1. Fit Naive baseline
    naive = NaiveForecaster()
    naive.fit(train_data)
    naive_preds = naive.predict(origin=cutoff_origin, horizon=horizon, calendar_df=cal_df)

    # 2. Fit LightGBM forecaster
    lgbm = LightGBMForecaster(
        objective="regression",
        n_estimators=60,
        learning_rate=0.08,
        random_state=42,
        horizon=horizon,
        min_history_days=30,
        sample_step_days=7,
    )
    lgbm.fit(train_data, calendar_df=cal_df)
    lgbm_preds = lgbm.predict(origin=cutoff_origin, horizon=horizon, calendar_df=cal_df)

    # Merge predictions with test actuals
    naive_eval = naive_preds.merge(test_actuals, on=["series_id", "date"], suffixes=("", "_true"))
    lgbm_eval = lgbm_preds.merge(test_actuals, on=["series_id", "date"], suffixes=("", "_true"))

    naive_wape = wape(naive_eval["y"].to_numpy(), naive_eval["yhat"].to_numpy())
    lgbm_wape = wape(lgbm_eval["y"].to_numpy(), lgbm_eval["yhat"].to_numpy())

    # LightGBM must beat naive on trending seasonal series
    assert lgbm_wape < naive_wape, f"LightGBM WAPE ({lgbm_wape:.4f}) >= Naive ({naive_wape:.4f})"
    # Also verify non-negative constraints
    assert (lgbm_preds["yhat"] >= 0.0).all()


def test_tweedie_objective_runs_and_predicts_non_negative() -> None:
    """Verify Tweedie objective trains properly and enforces yhat >= 0."""
    series_df, cal_df = _generate_synthetic_trending_seasonal_data(n_days=100, n_skus=2)
    cutoff_origin = datetime.date(2011, 3, 20)
    horizon = 14

    train_data = series_df[series_df["date"] <= cutoff_origin].copy()
    model = LightGBMForecaster(
        objective="tweedie",
        tweedie_variance_power=1.5,
        n_estimators=30,
        random_state=42,
        horizon=horizon,
    )
    model.fit(train_data, calendar_df=cal_df)
    preds = model.predict(origin=cutoff_origin, horizon=horizon, calendar_df=cal_df)

    assert not preds["yhat"].isna().any()
    assert (preds["yhat"] >= 0.0).all()
    assert len(preds) == 2 * horizon


def test_feature_importances() -> None:
    """Verify feature importances are computed and accessible."""
    series_df, cal_df = _generate_synthetic_trending_seasonal_data(n_days=100, n_skus=2)
    cutoff_origin = datetime.date(2011, 3, 20)

    model = LightGBMForecaster(n_estimators=30, random_state=42)
    model.fit(series_df[series_df["date"] <= cutoff_origin], calendar_df=cal_df)

    importances = model.get_feature_importances()
    assert not importances.empty
    assert "feature" in importances.columns
    assert "importance" in importances.columns
    assert (importances["importance"] >= 0.0).all()


def test_compare_objectives_helper() -> None:
    """Verify compare_objectives evaluates both objectives and picks a winner."""
    series_df, cal_df = _generate_synthetic_trending_seasonal_data(n_days=120, n_skus=2)

    best_model, report = compare_objectives(
        series_df=series_df,
        calendar_df=cal_df,
        horizon=14,
        n_folds=2,
        n_estimators=25,
        random_state=42,
    )

    assert isinstance(best_model, LightGBMForecaster)
    assert report["winner"] in ("tweedie", "regression")
    assert "tweedie_wape" in report
    assert "regression_wape" in report
    assert report["tweedie_wape"] >= 0.0
    assert report["regression_wape"] >= 0.0


def test_edge_cases_and_error_handling() -> None:
    """Verify error validations and edge cases."""
    # 1. Invalid objective
    with pytest.raises(ValueError, match="Unsupported objective"):
        LightGBMForecaster(objective="invalid_obj")

    # 2. Predict before fit
    unfitted = LightGBMForecaster()
    with pytest.raises(ValueError, match="must be fitted before predict"):
        unfitted.predict(origin=datetime.date(2011, 1, 1), horizon=7)

    # 3. Fit on empty history
    with pytest.raises(ValueError, match="Cannot fit LightGBMForecaster on empty"):
        unfitted.fit(pd.DataFrame())

    # 4. Feature importances before fit
    empty_fi = unfitted.get_feature_importances()
    assert empty_fi.empty

    # 5. Fit on history with insufficient span
    short_df = pd.DataFrame(
        {
            "series_id": [1],
            "product_id": [1],
            "date": [datetime.date(2011, 1, 1)],
            "y": [10.0],
            "price": [5.0],
            "is_holiday": [False],
            "abc_class": ["A"],
        }
    )
    with pytest.raises(ValueError, match="Insufficient historical span"):
        unfitted.fit(short_df)
