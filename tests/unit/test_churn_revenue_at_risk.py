"""Unit tests for revenue at risk calculations and customer analytics report (P3-07).

Implements TRD §5.3, PRD §7.4 (FR-CH-07), and IMPLEMENTATION_PLAN P3-07:
- Hand-computed / known inputs verification for revenue at risk formula:
    revenue_at_risk = max(0, p_churn * net_revenue_365d * (churn_window_days / 365))
- Edge cases: zero probability, zero revenue, negative net revenue, empty inputs.
- Segment and risk band breakdown summary computation.
- Generation of the markdown report `reports/churn_and_segments.md`.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scripts.report_churn_segments import (
    build_churn_and_segments_markdown,
    generate_churn_and_segments_report,
)

from retailpulse.ml.churn.revenue_at_risk import (
    compute_customer_revenue_at_risk,
    compute_revenue_at_risk_summary,
    compute_total_revenue_at_risk,
)

pytestmark = pytest.mark.unit


def test_compute_customer_revenue_at_risk_known_inputs() -> None:
    """Verify formula against hand-calculated known values."""
    # Customer 1: p=0.8, rev=1000.0, window=90 -> 800.0 * 90 / 365 = 197.26027397...
    # Customer 2: p=0.2, rev=500.0, window=90 -> 100.0 * 90 / 365 = 24.65753424...
    # Customer 3: p=1.0, rev=3650.0, window=365 -> 3650.0
    # Customer 4: p=0.5, rev=-200.0, window=90 -> 0.0 (floored at zero)
    # Customer 5: p=0.0, rev=2000.0, window=90 -> 0.0
    probs = [0.8, 0.2, 1.0, 0.5, 0.0]
    revenues = [1000.0, 500.0, 3650.0, -200.0, 2000.0]

    # Test individual 90-day window
    res_c1 = compute_customer_revenue_at_risk(0.8, 1000.0, churn_window_days=90)
    assert np.isclose(res_c1, 800.0 * 90.0 / 365.0, atol=1e-5)

    res_c2 = compute_customer_revenue_at_risk(0.2, 500.0, churn_window_days=90)
    assert np.isclose(res_c2, 100.0 * 90.0 / 365.0, atol=1e-5)

    # Test full-year window (365d)
    res_c3 = compute_customer_revenue_at_risk(1.0, 3650.0, churn_window_days=365)
    assert np.isclose(res_c3, 3650.0, atol=1e-5)

    # Test negative revenue floored to 0.0
    res_c4 = compute_customer_revenue_at_risk(0.5, -200.0, churn_window_days=90)
    assert res_c4 == 0.0

    # Test 0 probability
    res_c5 = compute_customer_revenue_at_risk(0.0, 2000.0, churn_window_days=90)
    assert res_c5 == 0.0

    # Vectorized array input
    res_vec = compute_customer_revenue_at_risk(probs[:2], revenues[:2], churn_window_days=90)
    expected_vec = np.array([800.0 * 90.0 / 365.0, 100.0 * 90.0 / 365.0])
    np.testing.assert_allclose(res_vec, expected_vec, rtol=1e-5)


def test_compute_total_revenue_at_risk() -> None:
    """Verify total sum over customer population."""
    probs = [0.8, 0.2]
    revenues = [1000.0, 500.0]
    total = compute_total_revenue_at_risk(probs, revenues, churn_window_days=90)
    expected = (800.0 * 90.0 / 365.0) + (100.0 * 90.0 / 365.0)
    assert np.isclose(total, expected, atol=1e-5)

    # Empty inputs
    assert compute_total_revenue_at_risk([], []) == 0.0


def test_compute_revenue_at_risk_summary() -> None:
    """Verify summary aggregation by risk band and segment."""
    data: dict[str, Any] = {
        "customer_id": [1, 2, 3, 4],
        "churn_probability": [0.80, 0.50, 0.10, 0.20],
        "net_revenue_365d": [2000.0, 1500.0, 4000.0, 1000.0],
        "risk_band": ["high", "medium", "low", "low"],
        "segment_label": ["Champions", "At Risk", "Champions", "Loyal"],
    }
    df = pd.DataFrame(data)

    summary = compute_revenue_at_risk_summary(df, churn_window_days=90)

    assert summary["n_customers"] == 4
    assert summary["total_net_revenue"] == 8500.0
    assert summary["mean_churn_probability"] == 0.40
    assert summary["expected_revenue_at_risk"] > 0.0
    assert 0.0 < summary["pct_revenue_at_risk"] < 100.0

    # Risk band breakdown
    rb_summary = summary["risk_band_breakdown"]
    assert "high" in rb_summary
    assert "medium" in rb_summary
    assert "low" in rb_summary
    assert rb_summary["high"]["n_customers"] == 1
    assert rb_summary["low"]["n_customers"] == 2

    # Segment breakdown
    seg_summary = summary["segment_breakdown"]
    assert "Champions" in seg_summary
    assert "At Risk" in seg_summary
    assert "Loyal" in seg_summary


def test_build_churn_and_segments_markdown(tmp_path: Path) -> None:
    """Verify report generation produces valid Markdown with all required sections."""
    df_customers = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "customer_key": [f"cust_{i}" for i in range(1, 6)],
            "segment_id": [1, 2, 1, 3, 2],
            "segment_label": ["Champions", "At Risk", "Champions", "Loyal", "At Risk"],
            "recency_days": [10, 120, 15, 45, 180],
            "frequency": [12, 2, 10, 5, 1],
            "monetary": [3000.0, 400.0, 2500.0, 1200.0, 150.0],
            "churn_probability": [0.08, 0.72, 0.12, 0.35, 0.85],
            "risk_band": ["low", "high", "low", "medium", "high"],
            "net_revenue_365d": [3000.0, 400.0, 2500.0, 1200.0, 150.0],
            "top_drivers": [
                [
                    {
                        "feature": "recency_days",
                        "value": 10,
                        "impact": -0.25,
                        "direction": "decreases_risk",
                    },
                    {
                        "feature": "orders_90d",
                        "value": 4,
                        "impact": -0.15,
                        "direction": "decreases_risk",
                    },
                    {
                        "feature": "tenure_days",
                        "value": 350,
                        "impact": -0.05,
                        "direction": "decreases_risk",
                    },
                ]
            ]
            * 5,
        }
    )

    profiles_df = pd.DataFrame(
        [
            {
                "segment_id": 1,
                "segment_label": "Champions",
                "n_customers": 2,
                "revenue_share": 0.758,
                "avg_recency_days": 12.5,
                "avg_frequency": 11.0,
                "avg_monetary": 2750.0,
            },
            {
                "segment_id": 2,
                "segment_label": "At Risk",
                "n_customers": 2,
                "revenue_share": 0.076,
                "avg_recency_days": 150.0,
                "avg_frequency": 1.5,
                "avg_monetary": 275.0,
            },
            {
                "segment_id": 3,
                "segment_label": "Loyal",
                "n_customers": 1,
                "revenue_share": 0.166,
                "avg_recency_days": 45.0,
                "avg_frequency": 5.0,
                "avg_monetary": 1200.0,
            },
        ]
    )

    cutoff = datetime.date(2011, 12, 1)
    md_content = build_churn_and_segments_markdown(
        customers_df=df_customers,
        profiles_df=profiles_df,
        data_as_of=cutoff,
        churn_window_days=90,
    )

    assert "# RetailPulse — Customer Segmentation & Churn Analytics Report" in md_content
    assert "Executive Summary" in md_content
    assert "Expected Revenue at Risk" in md_content
    assert "Customer Segmentation Profiles" in md_content
    assert "Cross-Tabulation: Customer Segments vs Churn Risk" in md_content
    assert "Champions" in md_content
    assert "At Risk" in md_content

    # Verify file generation
    out_file = tmp_path / "churn_and_segments.md"
    generate_churn_and_segments_report(
        customers_df=df_customers,
        profiles_df=profiles_df,
        data_as_of=cutoff,
        output_path=out_file,
    )
    assert out_file.is_file()
    assert out_file.stat().st_size > 0
