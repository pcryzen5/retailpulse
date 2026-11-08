"""Unit tests for the demand forecasting backtest reporting script (P2-12)."""

from __future__ import annotations

import datetime
from pathlib import Path

import pandas as pd
import pytest
from scripts.report_forecast import (
    build_markdown_report_content,
    execute_evaluation,
    generate_report_figures,
    prepare_evaluation_series,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_daily_sales() -> pd.DataFrame:
    """Create 2 years of synthetic daily sales across 3 products in A, B, C tiers."""
    dates = [datetime.date(2010, 1, 1) + datetime.timedelta(days=i) for i in range(700)]
    rows = []
    # Product 101: High volume (Class A)
    # Product 102: Medium volume (Class B)
    # Product 103: Low/intermittent volume (Class C)
    configs = [
        ("101", 100.0, 15.0),
        ("102", 20.0, 10.0),
        ("103", 2.0, 5.0),
    ]
    for pid, mean_qty, price in configs:
        for i, d in enumerate(dates):
            # Weekly pattern
            qty = int(mean_qty + (i % 7) * (mean_qty * 0.1))
            rows.append(
                {
                    "product_id": pid,
                    "date_key": d,
                    "units_sold": qty,
                    "revenue": float(qty * price),
                    "avg_price": price,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_evaluation_series(synthetic_daily_sales: pd.DataFrame) -> None:
    """Verify stratified series preparation for backtesting."""
    as_of = datetime.date(2011, 11, 30)
    series_df, cal_df, selected = prepare_evaluation_series(
        synthetic_daily_sales,
        as_of=as_of,
        top_n=3,
    )

    assert not series_df.empty
    assert not cal_df.empty
    assert len(selected) == 3
    assert "abc_class" in series_df.columns
    assert set(series_df["series_id"].unique()) == {"101", "102", "103"}


def test_execute_evaluation_and_figures(
    synthetic_daily_sales: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Verify model backtesting, interval calibration, and plot rendering."""
    as_of = datetime.date(2011, 11, 30)
    series_df, cal_df, selected = prepare_evaluation_series(
        synthetic_daily_sales,
        as_of=as_of,
        top_n=3,
    )

    models = ["naive", "seasonal_naive"]
    summaries, gate_res = execute_evaluation(
        models_to_run=models,
        series_df=series_df,
        cal_df=cal_df,
        n_folds=2,
        horizon=14,
    )

    assert "naive" in summaries
    assert "seasonal_naive" in summaries
    assert gate_res.selected_champion in ("naive", "seasonal_naive")

    # Generate figures
    fig_dir = tmp_path / "figures"
    fig_paths = generate_report_figures(
        summaries=summaries,
        champion_name=gate_res.selected_champion,
        output_dir=fig_dir,
    )

    assert (fig_dir / "forecast_wape_by_model.png").exists()
    assert (fig_dir / "forecast_horizon_decay.png").exists()
    assert (fig_dir / "forecast_abc_comparison.png").exists()
    assert (fig_dir / "forecast_sample_predictions.png").exists()

    # Compile report markdown
    report_md = build_markdown_report_content(
        summaries=summaries,
        gate_res=gate_res,
        data_as_of=as_of,
        n_series=len(selected),
        n_folds=2,
        horizon=14,
        fig_paths=fig_paths,
    )

    assert "# RetailPulse — Demand Forecasting Backtest & Champion Evaluation Report" in report_md
    assert "## 1. Executive Summary & Champion Gate Verdict" in report_md
    assert "## 2. Global Model Evaluation Matrix" in report_md
    assert "## 3. Breakdown by ABC Revenue Classification" in report_md
    assert "## 4. Error Trajectory Across Forecast Horizons" in report_md
    assert "## 6. Honest Technical Discussion & Qualitative Insights" in report_md
    assert "Seasonal Naive (7d)" in report_md
