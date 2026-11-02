"""Unit tests for residual-quantile prediction intervals.

Verifies P2-05 & TRD §5.1:
- Residual-quantile intervals per (model, ABC class, horizon bucket)
- Quantile resolution for nominal coverage (default 80%: q10 and q90)
- Constraint enforcement: 0 <= yhat_lower <= yhat <= yhat_upper
- Robust fallback hierarchy: (abc, horizon) -> horizon -> abc -> global
- Serialization & deserialization (to_dict / from_dict)
- VERIFY REQUIREMENT: Empirical coverage within tolerance on synthetic noise
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.forecasting.intervals import (
    ResidualQuantilesTable,
    apply_prediction_intervals,
    evaluate_interval_metrics,
    fit_residual_quantiles,
    map_horizon_to_bucket,
)


def test_map_horizon_to_bucket() -> None:
    """Test mapping 1-based horizon days to standard bucket strings."""
    assert map_horizon_to_bucket(1) == "1-7"
    assert map_horizon_to_bucket(7) == "1-7"
    assert map_horizon_to_bucket(8) == "8-14"
    assert map_horizon_to_bucket(14) == "8-14"
    assert map_horizon_to_bucket(15) == "15-28"
    assert map_horizon_to_bucket(28) == "15-28"
    assert map_horizon_to_bucket(30) == "15-28"


def test_quantiles_table_serialization() -> None:
    """Test round-trip serialization of ResidualQuantilesTable."""
    table = ResidualQuantilesTable(
        model_name="lightgbm",
        level=0.80,
        q_lower=0.10,
        q_upper=0.90,
        buckets={
            ("A", "1-7"): (-2.5, 3.0),
            ("B", "8-14"): (-5.0, 6.2),
        },
        horizon_fallback={"1-7": (-3.0, 3.5), "8-14": (-6.0, 7.0)},
        abc_fallback={"A": (-4.0, 4.0), "B": (-5.5, 5.5)},
        global_fallback=(-7.0, 8.0),
        sample_counts={"A|1-7": 150, "B|8-14": 85},
    )

    data = table.to_dict()
    restored = ResidualQuantilesTable.from_dict(data)

    assert restored.model_name == "lightgbm"
    assert restored.level == 0.80
    assert restored.q_lower == 0.10
    assert restored.q_upper == 0.90
    assert restored.buckets == {
        ("A", "1-7"): (-2.5, 3.0),
        ("B", "8-14"): (-5.0, 6.2),
    }
    assert restored.horizon_fallback == {"1-7": (-3.0, 3.5), "8-14": (-6.0, 7.0)}
    assert restored.abc_fallback == {"A": (-4.0, 4.0), "B": (-5.5, 5.5)}
    assert restored.global_fallback == (-7.0, 8.0)
    assert restored.sample_counts == {"A|1-7": 150, "B|8-14": 85}


def test_tiered_fallback_hierarchy() -> None:
    """Test lookup fallback: (abc, horizon) -> horizon -> abc -> global."""
    table = ResidualQuantilesTable(
        model_name="test_model",
        level=0.80,
        q_lower=0.10,
        q_upper=0.90,
        buckets={
            ("A", "1-7"): (-1.0, 1.0),
        },
        horizon_fallback={
            "8-14": (-2.0, 2.0),
        },
        abc_fallback={
            "B": (-3.0, 3.0),
        },
        global_fallback=(-10.0, 10.0),
    )

    # 1. Exact match
    assert table.lookup("A", "1-7") == (-1.0, 1.0)
    assert table.lookup("a", "1-7") == (-1.0, 1.0)  # case-insensitive ABC

    # 2. Horizon fallback (A, 8-14) not in buckets -> horizon "8-14"
    assert table.lookup("A", "8-14") == (-2.0, 2.0)

    # 3. ABC fallback (B, 15-28) not in buckets, "15-28" not in horizon_fb -> ABC "B"
    assert table.lookup("B", "15-28") == (-3.0, 3.0)

    # 4. Global fallback (C, 15-28) neither in buckets nor horizon nor abc fb
    assert table.lookup("C", "15-28") == (-10.0, 10.0)


def test_apply_prediction_intervals_db_constraints() -> None:
    """Ensure database check constraints 0 <= yhat_lower <= yhat <= yhat_upper hold."""
    table = ResidualQuantilesTable(
        model_name="test_model",
        level=0.80,
        q_lower=0.10,
        q_upper=0.90,
        buckets={
            ("A", "1-7"): (-15.0, 20.0),  # large negative lower, positive upper
            ("B", "1-7"): (5.0, -2.0),  # anomalous positive lower, negative upper
        },
        global_fallback=(-5.0, 5.0),
    )

    preds = pd.DataFrame(
        {
            "yhat": [10.0, 8.0, 0.0],
            "abc_class": ["A", "B", "A"],
            "horizon_bucket": ["1-7", "1-7", "1-7"],
        }
    )

    result = apply_prediction_intervals(preds, table)

    # Row 0: yhat=10.0, lower delta=-15.0 -> 10 + (-15) = -5 -> clamped to 0.0
    # upper delta=+20.0 -> 10 + 20 = 30.0
    assert result.loc[0, "yhat_lower"] == 0.0
    assert result.loc[0, "yhat"] == 10.0
    assert result.loc[0, "yhat_upper"] == 30.0

    # Row 1: yhat=8.0, lower delta=+5.0 -> 8 + 5 = 13.0 -> clamped to yhat (8.0)
    # upper delta=-2.0 -> 8 + (-2) = 6.0 -> clamped to yhat (8.0)
    assert result.loc[1, "yhat_lower"] == 8.0
    assert result.loc[1, "yhat"] == 8.0
    assert result.loc[1, "yhat_upper"] == 8.0

    # Row 2: yhat=0.0, lower clamped to 0.0, upper = 20.0
    assert result.loc[2, "yhat_lower"] == 0.0
    assert result.loc[2, "yhat_upper"] == 20.0

    # Check constraints on entire result
    assert (result["yhat_lower"] >= 0.0).all()
    assert (result["yhat_lower"] <= result["yhat"]).all()
    assert (result["yhat"] <= result["yhat_upper"]).all()
    assert (result["interval_level"] == 0.80).all()


def test_apply_prediction_intervals_empty_dataframe() -> None:
    """Empty dataframe should return empty dataframe with required interval columns."""
    table = ResidualQuantilesTable("m", 0.8, 0.1, 0.9)
    empty = pd.DataFrame(columns=["yhat", "abc_class"])
    res = apply_prediction_intervals(empty, table)
    assert res.empty
    assert "yhat_lower" in res.columns
    assert "yhat_upper" in res.columns
    assert "interval_level" in res.columns


def test_apply_prediction_intervals_missing_yhat() -> None:
    """Predictions missing 'yhat' must raise ValueError."""
    table = ResidualQuantilesTable("m", 0.8, 0.1, 0.9)
    bad_df = pd.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(ValueError, match="must contain 'yhat' column"):
        apply_prediction_intervals(bad_df, table)


def test_fit_residual_quantiles_validation() -> None:
    """Validation checks on fit_residual_quantiles."""
    # Invalid level
    with pytest.raises(ValueError, match="Coverage level must be in"):
        fit_residual_quantiles(pd.DataFrame({"y": [1], "yhat": [1]}), level=1.5)
    with pytest.raises(ValueError, match="Coverage level must be in"):
        fit_residual_quantiles(pd.DataFrame({"y": [1], "yhat": [1]}), level=0.0)

    # Empty df
    with pytest.raises(ValueError, match="Cannot fit residual quantiles on empty"):
        fit_residual_quantiles(pd.DataFrame())

    # Missing residual / y / yhat
    with pytest.raises(ValueError, match="must contain 'residual'"):
        fit_residual_quantiles(pd.DataFrame({"foo": [1, 2, 3]}))

    # Auto-calculates residual when y and yhat are provided, defaults missing abc_class and horizon
    df = pd.DataFrame({"y": [10.0] * 12, "yhat": [9.0] * 12})
    tbl = fit_residual_quantiles(df, min_samples=5)
    assert tbl.sample_counts["C|1-7"] == 12
    assert ("C", "1-7") in tbl.buckets

    # Predict with missing optional columns (defaults to C, 1-7)
    preds = pd.DataFrame({"yhat": [10.0, 20.0]})
    res = apply_prediction_intervals(preds, tbl)
    assert len(res) == 2
    assert "yhat_lower" in res.columns
    assert "yhat_upper" in res.columns


def test_fit_residual_quantiles_min_samples_fallback() -> None:
    """Buckets with sample count < min_samples should fall back to hierarchy."""
    data = []
    # Bucket A|1-7 has 5 samples (< min_samples=10)
    for _ in range(5):
        data.append({"residual": 2.0, "abc_class": "A", "horizon_day": 3})
    # Bucket B|1-7 has 15 samples (>= min_samples=10)
    for _ in range(15):
        data.append({"residual": -1.0, "abc_class": "B", "horizon_day": 5})

    df = pd.DataFrame(data)
    table = fit_residual_quantiles(df, level=0.80, min_samples=10)

    # A|1-7 was NOT added to buckets because count=5 < 10
    assert ("A", "1-7") not in table.buckets
    # B|1-7 WAS added to buckets because count=15 >= 10
    assert ("B", "1-7") in table.buckets

    # Sample counts reflect actual occurrences
    assert table.sample_counts["A|1-7"] == 5
    assert table.sample_counts["B|1-7"] == 15


def test_evaluate_interval_metrics() -> None:
    """Test calculation of interval evaluation metrics."""
    # Missing columns
    with pytest.raises(ValueError, match="Missing required columns"):
        evaluate_interval_metrics(pd.DataFrame({"y": [1.0]}))

    eval_df = pd.DataFrame(
        {
            "y": [10.0, 15.0, 5.0, 25.0],
            "yhat_lower": [8.0, 10.0, 6.0, 20.0],  # row 2: y=5 < lower=6 (miss)
            "yhat_upper": [12.0, 20.0, 14.0, 24.0],  # row 3: y=25 > upper=24 (miss)
        }
    )
    # y in [lower, upper]:
    # row 0: 10 in [8, 12] -> True
    # row 1: 15 in [10, 20] -> True
    # row 2: 5 in [6, 14] -> False
    # row 3: 25 in [20, 24] -> False
    # Coverage = 2/4 = 0.50

    metrics = evaluate_interval_metrics(eval_df)
    assert metrics["coverage"] == 0.50
    assert metrics["is_within_tolerance"] is False  # 0.50 < 0.65 threshold
    assert metrics["pinball_loss_q90"] >= 0.0
    assert metrics["pinball_loss_q10"] >= 0.0


def test_empirical_coverage_on_synthetic_noise() -> None:
    """VERIFY REQUIREMENT: Residual-quantile intervals achieve nominal 80% coverage.

    Generates synthetic sales observations y = yhat + eps where eps ~ N(0, sigma^2).
    Calibrates on train set and verifies empirical coverage is within [0.78, 0.82] on test set.
    """
    rng = np.random.default_rng(42)
    n_train = 6000
    n_test = 4000

    # True base demand yhat around 100
    yhat_train = rng.uniform(50.0, 150.0, size=n_train)
    sigma = 12.0
    noise_train = rng.normal(0.0, sigma, size=n_train)
    y_train = yhat_train + noise_train

    train_df = pd.DataFrame(
        {
            "y": y_train,
            "yhat": yhat_train,
            "residual": noise_train,
            "abc_class": rng.choice(["A", "B", "C"], size=n_train),
            "horizon_day": rng.integers(1, 29, size=n_train),
        }
    )

    # Fit quantiles with nominal 80% level
    quantiles_table = fit_residual_quantiles(
        train_df,
        level=0.80,
        min_samples=20,
        model_name="synthetic_champion",
    )

    # For Gaussian N(0, 12^2), q10 approx -15.378, q90 approx +15.378
    assert quantiles_table.level == 0.80
    assert quantiles_table.q_lower == pytest.approx(0.10)
    assert quantiles_table.q_upper == pytest.approx(0.90)

    # Generate out-of-sample test set with same noise distribution
    yhat_test = rng.uniform(50.0, 150.0, size=n_test)
    noise_test = rng.normal(0.0, sigma, size=n_test)
    y_test = yhat_test + noise_test

    test_df = pd.DataFrame(
        {
            "y": y_test,
            "yhat": yhat_test,
            "abc_class": rng.choice(["A", "B", "C"], size=n_test),
            "horizon_day": rng.integers(1, 29, size=n_test),
        }
    )

    intervals_df = apply_prediction_intervals(test_df, quantiles_table)

    # Validate database constraints
    assert (intervals_df["yhat_lower"] >= 0.0).all()
    assert (intervals_df["yhat_lower"] <= intervals_df["yhat"]).all()
    assert (intervals_df["yhat"] <= intervals_df["yhat_upper"]).all()

    # Evaluate coverage
    eval_results = evaluate_interval_metrics(intervals_df)
    empirical_cov = eval_results["coverage"]

    # Nominal 80% coverage on 4000 samples has std err = sqrt(0.8*0.2/4000) ~= 0.0063
    # A +/- 2.5% band [0.775, 0.825] comfortably bounds 4 standard errors
    assert 0.775 <= empirical_cov <= 0.825
    assert eval_results["is_within_tolerance"] is True
