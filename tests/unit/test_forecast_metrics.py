"""Unit tests for forecasting evaluation metrics and champion gate.

Verifies P2-02 & TRD §5.1:
- WAPE with zero-scale handling
- MASE and RMSSE with in-sample seasonal naive scale (m=7) and zero-scale protection
- Bias with zero-actuals handling
- sMAPE with symmetric zero-protection
- Coverage for prediction intervals
- Pinball loss for quantile forecasts (q=0.1, q=0.9)
- Scoped aggregation (global, abc, horizon, sku)
- Champion gate logic (|bias| <= 0.15, WAPE < baseline)
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.forecasting.metrics import (
    bias,
    check_champion_gate,
    compute_forecast_metrics_by_scope,
    coverage,
    mae,
    mase,
    pinball_loss,
    rmse,
    rmsse,
    smape,
    wape,
)


def test_wape_hand_computed() -> None:
    """Verify WAPE matches exact hand-calculated formula sum(|y - yhat|) / sum(y)."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # |e| = [2, 2, 3], sum(|e|) = 7.0, sum(y) = 60.0 -> 7 / 60
    expected = 7.0 / 60.0
    assert wape(y_true, y_pred) == pytest.approx(expected)


def test_wape_zero_scale_handling() -> None:
    """Verify WAPE zero-scale edge cases."""
    # When both actuals and preds are zero -> error is 0, return 0.0
    assert wape(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0

    # When actuals are zero but preds > 0 -> division by zero, returns NaN
    result = wape(np.array([0.0, 0.0]), np.array([5.0, 10.0]))
    assert math.isnan(result)


def test_bias_hand_computed() -> None:
    """Verify bias calculation matches sum(yhat - y) / sum(y)."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    # sum(yhat - y) = (2 - 2 + 3) = 3.0, sum(y) = 60.0 -> 3 / 60 = 0.05
    assert bias(y_true, y_pred) == pytest.approx(0.05)


def test_bias_zero_scale_handling() -> None:
    """Verify bias zero-scale edge cases."""
    assert bias(np.array([0.0, 0.0]), np.array([0.0, 0.0])) == 0.0
    assert math.isnan(bias(np.array([0.0, 0.0]), np.array([2.0, 3.0])))


def test_mae_and_rmse() -> None:
    """Verify basic MAE and RMSE."""
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([14.0, 17.0])
    # errors: [4, 3] -> MAE = 3.5, RMSE = sqrt((16 + 9)/2) = sqrt(12.5)
    assert mae(y_true, y_pred) == pytest.approx(3.5)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt(12.5))
    assert math.isnan(mae(np.array([]), np.array([])))
    assert math.isnan(rmse(np.array([]), np.array([])))


def test_mase_hand_computed() -> None:
    """Verify MASE against known in-sample seasonal naive baseline scale."""
    # 14 days in-sample history with weekly seasonality (m=7)
    # y_train: week 1 = [1..7], week 2 = [3..9]
    # |y_t - y_{t-7}| = 2.0 for all t in 8..14 -> scale = 2.0
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    y_true = np.array([10.0, 12.0])
    y_pred = np.array([12.0, 14.0])  # MAE = 2.0

    # MASE = 2.0 / 2.0 = 1.0
    assert mase(y_true, y_pred, y_train, seasonal_period=7) == pytest.approx(1.0)


def test_mase_edge_cases() -> None:
    """Verify MASE safely handles constant series, short history, or empty actuals."""
    y_train = np.array([5.0] * 14)
    y_true = np.array([5.0, 6.0])
    y_pred = np.array([5.0, 5.0])
    assert math.isnan(mase(y_true, y_pred, y_train, seasonal_period=7))

    # Short history <= seasonal_period
    assert math.isnan(mase(y_true, y_pred, np.array([1.0, 2.0]), seasonal_period=7))

    # Empty forecast
    y_train_ok = np.array([1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    assert math.isnan(mase(np.array([]), np.array([]), y_train_ok, seasonal_period=7))


def test_rmsse_hand_computed() -> None:
    """Verify RMSSE against known in-sample seasonal scale."""
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    # (y_t - y_{t-7})^2 = 4.0 -> mse_scale = 4.0
    y_true = np.array([10.0, 12.0])
    y_pred = np.array([12.0, 14.0])  # errors = 2, 2 -> MSE = 4.0

    # RMSSE = sqrt(4.0 / 4.0) = 1.0
    assert rmsse(y_true, y_pred, y_train, seasonal_period=7) == pytest.approx(1.0)


def test_rmsse_edge_cases() -> None:
    """Verify RMSSE edge cases (short history, constant series, empty inputs)."""
    assert math.isnan(
        rmsse(np.array([1.0]), np.array([1.0]), np.array([1.0, 2.0]), seasonal_period=7)
    )
    assert math.isnan(
        rmsse(np.array([1.0]), np.array([1.0]), np.array([5.0] * 14), seasonal_period=7)
    )
    y_train_ok = np.array([1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    assert math.isnan(rmsse(np.array([]), np.array([]), y_train_ok, seasonal_period=7))


def test_smape_hand_computed() -> None:
    """Verify sMAPE with symmetric scale and empty inputs."""
    y_true = np.array([10.0, 0.0])
    y_pred = np.array([20.0, 0.0])
    # Term 1: 2 * |10 - 20| / (10 + 20) = 20 / 30 = 2/3
    # Term 2: 0 / 0 evaluated as 0.0
    # Average across 2 elements is 1/3
    assert smape(y_true, y_pred) == pytest.approx(1.0 / 3.0)
    assert math.isnan(smape(np.array([]), np.array([])))


def test_coverage_hand_computed() -> None:
    """Verify coverage correctly evaluates interval containment."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_lower = np.array([9.0, 19.0, 31.0, 35.0])
    y_upper = np.array([11.0, 21.0, 35.0, 45.0])
    # Containment: 10 in [9,11], 20 in [19,21], 30 not in [31,35], 40 in [35,45]
    # Expected rate: 3 / 4 = 0.75
    assert coverage(y_true, y_lower, y_upper) == pytest.approx(0.75)
    assert math.isnan(coverage(np.array([]), np.array([]), np.array([])))


def test_pinball_loss_hand_computed() -> None:
    """Verify pinball loss for over- and under-prediction."""
    # Under-prediction (y > yhat): e = 10 - 8 = 2
    # For q=0.9: loss = 0.9 * 2 = 1.8
    assert pinball_loss(np.array([10.0]), np.array([8.0]), quantile=0.9) == pytest.approx(1.8)

    # Over-prediction (y < yhat): e = 10 - 12 = -2
    # For q=0.9: loss = (1 - 0.9) * 2 = 0.2
    assert pinball_loss(np.array([10.0]), np.array([12.0]), quantile=0.9) == pytest.approx(0.2)

    # Empty array
    assert math.isnan(pinball_loss(np.array([]), np.array([]), quantile=0.5))

    # Invalid quantile
    with pytest.raises(ValueError, match="quantile must be strictly in"):
        pinball_loss(np.array([1.0]), np.array([1.0]), quantile=1.5)


def test_check_champion_gate() -> None:
    """Verify champion gate criteria: WAPE < baseline and |bias| <= 0.15."""
    # Passing candidate
    passed, warnings, reason = check_champion_gate(
        candidate_wape=0.25,
        baseline_wape=0.30,
        candidate_bias=0.05,
        candidate_coverage=0.80,
    )
    assert passed is True
    assert len(warnings) == 0
    assert "beats baseline" in reason

    # Fails WAPE gate
    passed, _, reason = check_champion_gate(
        candidate_wape=0.35,
        baseline_wape=0.30,
        candidate_bias=0.05,
    )
    assert passed is False
    assert "WAPE" in reason

    # Fails Bias gate
    passed, _, reason = check_champion_gate(
        candidate_wape=0.25,
        baseline_wape=0.30,
        candidate_bias=0.18,  # > 0.15
    )
    assert passed is False
    assert "bias" in reason

    # Passes gate but raises coverage warning (outside [0.65, 0.95])
    passed, warnings, _ = check_champion_gate(
        candidate_wape=0.25,
        baseline_wape=0.30,
        candidate_bias=0.05,
        candidate_coverage=0.55,
    )
    assert passed is True
    assert len(warnings) == 1
    assert "coverage" in warnings[0].lower()

    # NaN candidate or baseline
    p_nan1, _, r_nan1 = check_champion_gate(float("nan"), 0.30, 0.05)
    assert p_nan1 is False
    assert "Candidate WAPE is NaN" in r_nan1

    p_nan2, _, r_nan2 = check_champion_gate(0.25, float("nan"), 0.05)
    assert p_nan2 is False
    assert "Baseline WAPE is NaN" in r_nan2


def test_compute_forecast_metrics_by_scope() -> None:
    """Verify multi-scope metrics computation (global, abc, horizon, sku) with in-sample history."""
    rows = []
    for sid, abc in [(1, "A"), (2, "B")]:
        for h in range(1, 29):
            y = 10.0
            yhat = 12.0  # error = 2
            rows.append(
                {
                    "series_id": sid,
                    "product_id": sid,
                    "horizon_day": h,
                    "abc_class": abc,
                    "y": y,
                    "yhat": yhat,
                    "yhat_lower": 8.0,
                    "yhat_upper": 15.0,
                }
            )
    eval_df = pd.DataFrame(rows)

    # Provide in-sample training history
    y_train = np.array([1, 2, 3, 4, 5, 6, 7, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    in_sample = {1: y_train, 2: y_train}

    metrics_df = compute_forecast_metrics_by_scope(eval_df, in_sample_history=in_sample)
    assert not metrics_df.empty

    for col in ["scope", "scope_key", "metric", "value", "n_series"]:
        assert col in metrics_df.columns

    scopes = set(metrics_df["scope"])
    assert "global" in scopes
    assert "abc" in scopes
    assert "horizon" in scopes
    assert "sku" in scopes

    # Check global WAPE = 2 / 10 = 0.20
    global_wape = metrics_df[
        (metrics_df["scope"] == "global")
        & (metrics_df["scope_key"] == "all")
        & (metrics_df["metric"] == "wape")
    ]["value"].iloc[0]
    assert global_wape == pytest.approx(0.20)

    # Check SKU scope has mase and rmsse
    sku_metrics = set(metrics_df[metrics_df["scope"] == "sku"]["metric"])
    assert "mase" in sku_metrics
    assert "rmsse" in sku_metrics

    # Check missing columns raise ValueError
    with pytest.raises(ValueError, match="Missing required columns"):
        compute_forecast_metrics_by_scope(pd.DataFrame([{"invalid": 1}]))
