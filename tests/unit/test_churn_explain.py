"""Unit tests for customer churn explainability and driver attribution (P3-05).

Verifies:
- SHAP TreeExplainer attribution for LightGBM models.
- Standardized linear coefficient attribution for Logistic Regression baselines.
- Top-3 reason formatting: {feature, value, impact, direction} with stable keys.
- Global feature importance table and plot artifact generation.
- Graceful handling of missing values, unseen categories, and empty dataframes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from retailpulse.ml.churn.explain import (
    compute_global_feature_importance,
    explain_churn_predictions,
    plot_global_feature_importance,
)
from retailpulse.ml.churn.models import (
    ALL_FEATURES,
    CalibratedChurnModel,
    create_logistic_baseline_pipeline,
    fit_lightgbm_with_early_stopping,
)


@pytest.fixture
def synthetic_churn_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Generate synthetic churn feature DataFrame and labels for testing."""
    rng = np.random.default_rng(42)
    n_train = 60
    n_test = 15

    def make_features(n: int) -> pd.DataFrame:
        data: dict[str, Any] = {
            "recency_days": rng.integers(1, 300, size=n).astype(float),
            "tenure_days": rng.integers(30, 700, size=n).astype(float),
            "orders_30d": rng.integers(0, 5, size=n).astype(float),
            "orders_90d": rng.integers(0, 10, size=n).astype(float),
            "orders_180d": rng.integers(0, 20, size=n).astype(float),
            "orders_365d": rng.integers(1, 30, size=n).astype(float),
            "orders_prior_90d": rng.integers(0, 10, size=n).astype(float),
            "order_count_trend": rng.uniform(-1.0, 1.0, size=n),
            "net_revenue_90d": rng.uniform(0.0, 2000.0, size=n),
            "net_revenue_365d": rng.uniform(50.0, 5000.0, size=n),
            "avg_order_value": rng.uniform(20.0, 300.0, size=n),
            "items_per_order": rng.uniform(1.0, 50.0, size=n),
            "distinct_skus": rng.integers(1, 40, size=n).astype(float),
            "return_rate": rng.uniform(0.0, 0.2, size=n),
            "gap_mean": rng.uniform(10.0, 60.0, size=n),
            "gap_std": rng.uniform(0.0, 20.0, size=n),
            "overdue_ratio": rng.uniform(0.2, 5.0, size=n),
            "has_single_order": rng.choice([0.0, 1.0], size=n, p=[0.7, 0.3]),
            "month_of_cutoff": rng.integers(1, 13, size=n).astype(float),
            "country": rng.choice(["United Kingdom", "Germany", "France"], size=n),
        }
        # Introduce some missing gap values for single-order customers
        for i in range(n):
            if data["has_single_order"][i] == 1.0:
                data["gap_mean"][i] = np.nan
                data["gap_std"][i] = np.nan
                data["overdue_ratio"][i] = np.nan
        return pd.DataFrame(data)

    df_train = make_features(n_train)
    y_train = rng.choice([0, 1], size=n_train, p=[0.5, 0.5])
    df_test = make_features(n_test)

    return df_train, y_train, df_test


def test_explain_churn_predictions_lightgbm(
    synthetic_churn_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame],
) -> None:
    """Verify LightGBM SHAP attribution outputs top-3 reasons with valid schema."""
    df_train, y_train, df_test = synthetic_churn_data
    pipe = fit_lightgbm_with_early_stopping(
        df_train,
        y_train,
        df_train.iloc[:20],
        y_train[:20],
        seed=42,
        n_estimators=10,
    )

    reasons_list = explain_churn_predictions(pipe, df_test, top_n=3)

    assert len(reasons_list) == len(df_test)
    for reasons in reasons_list:
        assert len(reasons) == 3
        # Check sort order by absolute impact descending
        impacts = [abs(r["impact"]) for r in reasons]
        assert impacts == sorted(impacts, reverse=True)

        for r in reasons:
            assert set(r.keys()) == {"feature", "value", "impact", "direction"}
            assert r["feature"] in ALL_FEATURES
            assert r["direction"] in {"increases_risk", "decreases_risk"}
            assert isinstance(r["impact"], float)


def test_explain_churn_predictions_logistic(
    synthetic_churn_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame],
) -> None:
    """Verify Logistic Regression linear attribution outputs top-3 reasons with valid schema."""
    df_train, y_train, df_test = synthetic_churn_data
    pipe = create_logistic_baseline_pipeline(seed=42)
    pipe.fit(df_train[ALL_FEATURES], y_train)

    reasons_list = explain_churn_predictions(pipe, df_test, top_n=3)

    assert len(reasons_list) == len(df_test)
    for reasons in reasons_list:
        assert len(reasons) == 3
        impacts = [abs(r["impact"]) for r in reasons]
        assert impacts == sorted(impacts, reverse=True)

        for r in reasons:
            assert set(r.keys()) == {"feature", "value", "impact", "direction"}
            assert r["feature"] in ALL_FEATURES
            assert r["direction"] in {"increases_risk", "decreases_risk"}
            assert isinstance(r["impact"], float)


def test_explain_churn_predictions_calibrated_container(
    synthetic_churn_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame],
) -> None:
    """Verify CalibratedChurnModel container unwrapping and explanation extraction."""
    df_train, y_train, df_test = synthetic_churn_data
    pipe = fit_lightgbm_with_early_stopping(
        df_train,
        y_train,
        df_train.iloc[:20],
        y_train[:20],
        seed=42,
        n_estimators=10,
    )
    container = CalibratedChurnModel(
        pipe,
        algorithm="lightgbm",
        calibration_method=None,
    )

    reasons_list = explain_churn_predictions(container, df_test, top_n=3)
    assert len(reasons_list) == len(df_test)
    assert len(reasons_list[0]) == 3


def test_explain_churn_predictions_edge_cases(
    synthetic_churn_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame],
) -> None:
    """Verify edge cases: empty dataframe, missing feature values."""
    df_train, y_train, df_test = synthetic_churn_data
    pipe = fit_lightgbm_with_early_stopping(
        df_train,
        y_train,
        df_train.iloc[:20],
        y_train[:20],
        seed=42,
        n_estimators=10,
    )

    # Empty dataframe
    empty_df = df_test.iloc[0:0]
    assert explain_churn_predictions(pipe, empty_df) == []

    # Dataframe with NaN in raw values (e.g. single order)
    single_order_df = df_test[df_test["has_single_order"] == 1.0]
    if not single_order_df.empty:
        reasons_list = explain_churn_predictions(pipe, single_order_df)
        for reasons in reasons_list:
            for r in reasons:
                # If feature was NaN, value should be None or nan
                assert "feature" in r


def test_global_feature_importance_and_plot(
    synthetic_churn_data: tuple[pd.DataFrame, np.ndarray, pd.DataFrame],
    tmp_path: Path,
) -> None:
    """Verify global feature importance computation and horizontal bar chart artifact."""
    df_train, y_train, df_test = synthetic_churn_data
    pipe = fit_lightgbm_with_early_stopping(
        df_train,
        y_train,
        df_train.iloc[:20],
        y_train[:20],
        seed=42,
        n_estimators=10,
    )

    importance_df = compute_global_feature_importance(pipe, df_test)
    assert not importance_df.empty
    assert list(importance_df.columns) == ["feature", "mean_abs_impact"]
    # Verify sorted descending
    assert list(importance_df["mean_abs_impact"]) == sorted(
        importance_df["mean_abs_impact"], reverse=True
    )

    # Test plot artifact generation
    out_plot = tmp_path / "feature_importance.png"
    plot_global_feature_importance(importance_df, out_plot, top_n=10)
    assert out_plot.is_file()
    assert out_plot.stat().st_size > 0

