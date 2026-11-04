"""Unit tests for Prophet forecaster.

Verifies P2-07 & TRD §5.1:
- Conformance to Forecaster protocol
- Parallel fitting across series via joblib
- GB holidays inclusion
- Automatic fallback to Croston-SBA for ineligible series (< 90 days history or < 20 sales days)
- Non-negative prediction constraint (yhat >= 0.0)
- Smoke test on 3 series (marked @pytest.mark.slow)
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest
from prophet import Prophet

from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.prophet_model import ProphetForecaster, check_series_eligibility


def _create_series_data(
    series_id: int | str,
    n_days: int,
    sales_frequency: int = 1,  # sale every N days
    base_demand: float = 20.0,
    start_date: datetime.date = datetime.date(2011, 1, 1),
) -> pd.DataFrame:
    """Helper to generate a synthetic sales history series."""
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]
    y_vals: list[float] = []
    for i in range(n_days):
        if i % sales_frequency == 0:
            y_vals.append(base_demand + (i % 7) * 2.0)
        else:
            y_vals.append(0.0)

    return pd.DataFrame(
        {
            "series_id": [series_id] * n_days,
            "product_id": [series_id] * n_days,
            "date": dates,
            "y": y_vals,
            "price": [15.0] * n_days,
            "is_holiday": [False] * n_days,
            "abc_class": ["A"] * n_days,
        }
    )


def test_protocol_conformance() -> None:
    """Verify ProphetForecaster conforms to Forecaster Protocol."""
    model = ProphetForecaster(n_jobs=1)
    assert isinstance(model, Forecaster)
    assert model.name == "prophet"
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_check_series_eligibility() -> None:
    """Verify eligibility logic: >= 90 days span and >= 20 non-zero sales days."""
    # 0. Empty series
    is_elig, reason = check_series_eligibility(pd.DataFrame())
    assert is_elig is False
    assert reason == "empty_series"

    # 1. Eligible series: 120 days, sales every day (120 sales days >= 20)
    eligible_df = _create_series_data(1, n_days=120, sales_frequency=1)
    is_elig, reason = check_series_eligibility(
        eligible_df,
        min_history_days=90,
        min_sales_days=20,
    )
    assert is_elig is True
    assert reason is None

    # 2. Ineligible: span too short (< 90 days)
    short_df = _create_series_data(2, n_days=50, sales_frequency=1)
    is_elig, reason = check_series_eligibility(
        short_df,
        min_history_days=90,
        min_sales_days=20,
    )
    assert is_elig is False
    assert reason == "history_too_short"

    # 3. Ineligible: sparse sales (< 20 non-zero days in 120 days)
    sparse_df = _create_series_data(3, n_days=120, sales_frequency=10)  # 12 sales days < 20
    is_elig, reason = check_series_eligibility(
        sparse_df,
        min_history_days=90,
        min_sales_days=20,
    )
    assert is_elig is False
    assert reason == "too_few_sales_days"


def test_prophet_fit_exception_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that if Prophet fit raises an exception, it falls back to Croston-SBA."""

    def mock_fit(self: Prophet, df: pd.DataFrame) -> None:
        raise RuntimeError("Mocked Stan fitting error")

    monkeypatch.setattr(Prophet, "fit", mock_fit)

    eligible_df = _create_series_data(10, n_days=100, sales_frequency=1)
    cutoff = datetime.date(2011, 4, 1)

    model = ProphetForecaster(n_jobs=1, min_history_days=90, min_sales_days=20)
    model.fit(eligible_df)

    assert model.model_types[10] == "croston_sba"
    assert "Mocked Stan fitting error" in model.ineligibility_reasons[10]

    preds = model.predict(origin=cutoff, horizon=7)
    assert len(preds) == 7
    assert (preds["model_type"] == "croston_sba").all()


def test_prophet_zero_horizon() -> None:
    """Verify zero horizon returns empty DataFrame."""
    short_series = _create_series_data(1, n_days=40, sales_frequency=2)
    cutoff = datetime.date(2011, 2, 9)

    model = ProphetForecaster(n_jobs=1, min_history_days=90, min_sales_days=20)
    model.fit(short_series)

    preds = model.predict(origin=cutoff, horizon=0)
    assert preds.empty


def test_prophet_fallback_to_croston_sba() -> None:
    """Verify ineligible series automatically fall back to Croston-SBA forecaster."""
    # Series 1 is short (ineligible)
    short_series = _create_series_data(1, n_days=40, sales_frequency=2)
    cutoff = datetime.date(2011, 2, 9)

    model = ProphetForecaster(n_jobs=1, min_history_days=90, min_sales_days=20)
    model.fit(short_series)

    assert model.model_types.get(1) == "croston_sba"
    assert "history_too_short" in model.ineligibility_reasons.get(1, "")

    preds = model.predict(origin=cutoff, horizon=7)
    assert len(preds) == 7
    assert (preds["series_id"] == 1).all()
    assert (preds["yhat"] >= 0.0).all()
    assert (preds["model_type"] == "croston_sba").all()


def test_prophet_empty_dataframe_handling() -> None:
    """Verify empty input handling in fit and predict."""
    model = ProphetForecaster(n_jobs=1)
    with pytest.raises(ValueError, match="Cannot fit ProphetForecaster on empty"):
        model.fit(pd.DataFrame())

    unfitted = ProphetForecaster(n_jobs=1)
    with pytest.raises(ValueError, match="must be fitted before predict"):
        unfitted.predict(origin=datetime.date(2011, 1, 1), horizon=7)


@pytest.mark.slow
def test_prophet_smoke_test_three_series() -> None:
    """VERIFY REQUIREMENT: Smoke test ProphetForecaster on 3 series.

    - Series 1: Eligible regular demand (fits with Prophet)
    - Series 2: Ineligible short history (falls back to Croston-SBA)
    - Series 3: Eligible weekly periodic demand with weekend peaks (fits with Prophet)
    Verifies parallel fit, GB holidays, non-negative predictions, and fallback dispatch.
    """
    # Series 1: 120 days, regular sales
    s1 = _create_series_data(
        series_id="SKU_REGULAR", n_days=120, sales_frequency=1, base_demand=25.0
    )

    # Series 2: 45 days (ineligible, history < 90 days)
    s2 = _create_series_data(series_id="SKU_SHORT", n_days=45, sales_frequency=2, base_demand=10.0)

    # Series 3: 130 days, cyclic sales
    s3 = _create_series_data(
        series_id="SKU_CYCLIC", n_days=130, sales_frequency=1, base_demand=50.0
    )

    combined_history = pd.concat([s1, s2, s3], ignore_index=True)
    cutoff = datetime.date(2011, 5, 1)
    train_history = combined_history[combined_history["date"] <= cutoff].copy()
    horizon = 14

    # Fit in parallel with 2 jobs and GB holidays
    forecaster = ProphetForecaster(
        n_jobs=2,
        country_holidays="GB",
        weekly_seasonality=True,
        yearly_seasonality=False,
        min_history_days=90,
        min_sales_days=20,
    )
    forecaster.fit(train_history)

    # Verify model dispatch
    assert forecaster.model_types["SKU_REGULAR"] == "prophet"
    assert forecaster.model_types["SKU_SHORT"] == "croston_sba"
    assert forecaster.model_types["SKU_CYCLIC"] == "prophet"

    # Generate predictions
    preds = forecaster.predict(origin=cutoff, horizon=horizon)

    assert not preds.empty
    assert len(preds) == 3 * horizon
    assert set(preds["series_id"]) == {"SKU_REGULAR", "SKU_SHORT", "SKU_CYCLIC"}
    assert (preds["yhat"] >= 0.0).all()

    # Dates are strictly after cutoff
    assert (preds["date"] > cutoff).all()
    assert preds["date"].min() == cutoff + datetime.timedelta(days=1)
    assert preds["date"].max() == cutoff + datetime.timedelta(days=horizon)

    # Prophet native intervals present and non-negative
    prophet_rows = preds[preds["model_type"] == "prophet"]
    assert "prophet_yhat_lower" in prophet_rows.columns
    assert "prophet_yhat_upper" in prophet_rows.columns
    assert (prophet_rows["prophet_yhat_lower"] >= 0.0).all()
