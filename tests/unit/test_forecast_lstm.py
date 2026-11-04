"""Unit tests for LSTM PyTorch demand forecaster.

Verifies P2-08 & TRD §5.1:
- Conformance to Forecaster protocol
- PyTorch CPU execution
- Windowing: 56-day input window to 28-day target output
- Scaled log1p(units), day-of-week sin/cos, holiday, price ratio, and static SKU embedding
- Huber loss on log1p scale with expm1 clipping (yhat >= 0.0)
- Early stopping mechanism
- VERIFY REQUIREMENT: Tiny-data test (training loss decreases; deterministic output with same seed)
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.forecasting.base import Forecaster
from retailpulse.ml.forecasting.lstm_model import LSTMForecaster, build_lstm_windows


def _make_synthetic_lstm_data(
    n_days: int = 120,
    n_skus: int = 2,
    start_date: datetime.date = datetime.date(2011, 1, 1),
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic sales data suitable for 56-day windowing."""
    rng = np.random.default_rng(random_seed)
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_days)]

    rows: list[dict[str, object]] = []
    for sku in range(1, n_skus + 1):
        price = 10.0 * sku
        for t, d in enumerate(dates):
            dow = d.isocalendar()[2] - 1
            # Periodic weekly demand + linear trend
            base = 15.0 + 5.0 * (dow % 3) + 0.1 * t
            noise = rng.normal(0.0, 1.0)
            y_val = max(0.0, base + noise)

            rows.append(
                {
                    "series_id": sku,
                    "product_id": sku,
                    "date": d,
                    "y": y_val,
                    "price": price,
                    "is_holiday": (dow == 6),
                    "abc_class": "A",
                }
            )

    return pd.DataFrame(rows)


def test_protocol_conformance() -> None:
    """Verify LSTMForecaster conforms to Forecaster Protocol."""
    model = LSTMForecaster(random_seed=42)
    assert isinstance(model, Forecaster)
    assert model.name == "lstm"
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")


def test_windowing_shape_and_features() -> None:
    """Verify window extraction produces correct tensor shapes and features."""
    df = _make_synthetic_lstm_data(n_days=90, n_skus=1)
    sku_to_idx = {1: 0}

    windows = build_lstm_windows(
        df,
        sku_to_idx=sku_to_idx,
        window_size=56,
        horizon=28,
        stride_days=1,
    )

    assert windows.x_seq.ndim == 3
    # Shape: (N, window_size=56, num_features=5)
    assert windows.x_seq.shape[1] == 56
    assert windows.x_seq.shape[2] == 5
    # Target shape: (N, horizon=28)
    assert windows.y_target.shape[1] == 28
    # SKU indices
    assert windows.sku_indices.shape[0] == windows.x_seq.shape[0]


def test_tiny_data_loss_decreases() -> None:
    """VERIFY REQUIREMENT: Training loss decreases over epochs on tiny synthetic dataset."""
    df = _make_synthetic_lstm_data(n_days=100, n_skus=2, random_seed=42)

    model = LSTMForecaster(
        window_size=56,
        horizon=28,
        hidden_size=32,
        embedding_dim=8,
        num_layers=1,
        learning_rate=0.01,
        batch_size=32,
        max_epochs=15,
        early_stopping_patience=10,
        random_seed=42,
    )
    model.fit(df)

    history = model.training_history
    assert len(history) > 1
    # Check that final training loss is lower than initial epoch loss
    initial_loss = history[0]["train_loss"]
    final_loss = history[-1]["train_loss"]
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"


def test_determinism_same_seed_gives_identical_output() -> None:
    """VERIFY REQUIREMENT: Two LSTM instances with same seed produce identical predictions."""
    df = _make_synthetic_lstm_data(n_days=110, n_skus=2, random_seed=42)
    cutoff = datetime.date(2011, 4, 1)
    horizon = 28

    train_data = df[df["date"] <= cutoff].copy()

    model1 = LSTMForecaster(
        window_size=56,
        horizon=horizon,
        hidden_size=32,
        max_epochs=5,
        random_seed=123,
    )
    model1.fit(train_data)
    preds1 = model1.predict(origin=cutoff, horizon=horizon)

    model2 = LSTMForecaster(
        window_size=56,
        horizon=horizon,
        hidden_size=32,
        max_epochs=5,
        random_seed=123,
    )
    model2.fit(train_data)
    preds2 = model2.predict(origin=cutoff, horizon=horizon)

    assert not preds1.empty
    assert not preds2.empty
    assert len(preds1) == len(preds2)
    np.testing.assert_allclose(preds1["yhat"].to_numpy(), preds2["yhat"].to_numpy(), atol=1e-5)


def test_non_negative_predictions_constraint() -> None:
    """Verify LSTM forecaster outputs satisfy yhat >= 0.0 constraint."""
    df = _make_synthetic_lstm_data(n_days=100, n_skus=2, random_seed=42)
    cutoff = datetime.date(2011, 3, 20)

    model = LSTMForecaster(window_size=56, horizon=14, max_epochs=3, random_seed=42)
    model.fit(df[df["date"] <= cutoff])
    preds = model.predict(origin=cutoff, horizon=14)

    assert not preds["yhat"].isna().any()
    assert (preds["yhat"] >= 0.0).all()
    assert (preds["date"] > cutoff).all()
    assert len(preds) == 2 * 14


def test_lstm_error_handling_and_edge_cases() -> None:
    """Verify input validation and error handling."""
    unfitted = LSTMForecaster()
    with pytest.raises(ValueError, match="must be fitted before predict"):
        unfitted.predict(origin=datetime.date(2011, 1, 1), horizon=7)

    with pytest.raises(ValueError, match="Cannot fit LSTMForecaster on empty"):
        unfitted.fit(pd.DataFrame())

    # Short history (< window_size + horizon)
    short_df = pd.DataFrame(
        {
            "series_id": [1],
            "product_id": [1],
            "date": [datetime.date(2011, 1, 1)],
            "y": [5.0],
            "price": [10.0],
            "is_holiday": [False],
            "abc_class": ["A"],
        }
    )
    with pytest.raises(ValueError, match="Insufficient history span"):
        unfitted.fit(short_df)
