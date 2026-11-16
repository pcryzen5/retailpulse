"""Expected revenue at risk calculation and customer analytics aggregation.

Implements P3-07, TRD §5.3 & PRD §7.4 (FR-CH-07):
- Expected revenue at risk formula:
    revenue_at_risk = max(0, p_churn * net_revenue_365d * (churn_window_days / 365.0))
- Total expected revenue at risk summed across active customers.
- Breakdowns by canonical risk band ('high', 'medium', 'low') and customer segments.
- Cross-tabulation matrices between customer segments and churn risk tiers.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, overload

import numpy as np
import numpy.typing as npt
import pandas as pd

from retailpulse.logging import get_logger

logger = get_logger(__name__)


@overload
def compute_customer_revenue_at_risk(
    p_churn: float,
    net_revenue_365d: float,
    *,
    churn_window_days: int = 90,
) -> float: ...


@overload
def compute_customer_revenue_at_risk(
    p_churn: Sequence[float] | npt.NDArray[Any] | pd.Series,
    net_revenue_365d: Sequence[float] | npt.NDArray[Any] | pd.Series,
    *,
    churn_window_days: int = 90,
) -> npt.NDArray[np.float64]: ...


def compute_customer_revenue_at_risk(
    p_churn: float | Sequence[float] | npt.NDArray[Any] | pd.Series,
    net_revenue_365d: float | Sequence[float] | npt.NDArray[Any] | pd.Series,
    *,
    churn_window_days: int = 90,
) -> float | npt.NDArray[np.float64]:
    """Calculate expected revenue at risk per customer.

    Formula (TRD §5.3):
        revenue_at_risk = max(0.0, p_churn * trailing_12m_net_revenue * (churn_window_days / 365.0))

    Args:
        p_churn: Churn probability in [0, 1] (scalar or sequence).
        net_revenue_365d: Trailing 12-month net revenue (scalar or sequence).
        churn_window_days: Churn prediction window in days (default 90).

    Returns:
        Expected revenue at risk as float (for scalar input) or numpy array (for sequence).
    """
    scale = float(churn_window_days) / 365.0

    if isinstance(p_churn, (int, float)) and isinstance(net_revenue_365d, (int, float)):
        val = float(p_churn) * float(net_revenue_365d) * scale
        return max(0.0, val)

    p_arr = np.asarray(p_churn, dtype=np.float64)
    r_arr = np.asarray(net_revenue_365d, dtype=np.float64)

    at_risk = p_arr * r_arr * scale
    # Floor negative revenue at zero so risk is non-negative
    return np.maximum(0.0, at_risk)


def compute_total_revenue_at_risk(
    p_churn: Sequence[float] | npt.NDArray[Any] | pd.Series,
    net_revenue_365d: Sequence[float] | npt.NDArray[Any] | pd.Series,
    *,
    churn_window_days: int = 90,
) -> float:
    """Compute total expected revenue at risk summed across all active customers.

    Args:
        p_churn: Churn probabilities for active customers.
        net_revenue_365d: Trailing 12-month net revenue for active customers.
        churn_window_days: Observation window in days (default 90).

    Returns:
        Sum of expected revenue at risk as a float.
    """
    if len(p_churn) == 0:
        return 0.0

    at_risk_arr = compute_customer_revenue_at_risk(
        p_churn,
        net_revenue_365d,
        churn_window_days=churn_window_days,
    )
    return float(np.sum(at_risk_arr))


def compute_revenue_at_risk_summary(
    df: pd.DataFrame,
    *,
    prob_col: str = "churn_probability",
    revenue_col: str = "net_revenue_365d",
    segment_col: str = "segment_label",
    risk_band_col: str = "risk_band",
    churn_window_days: int = 90,
) -> dict[str, Any]:
    """Compute consolidated revenue at risk summary and dimensional breakdowns.

    Args:
        df: Customer DataFrame containing probabilities, revenues, and optional segments/bands.
        prob_col: Column name containing churn probabilities.
        revenue_col: Column name containing trailing 12m net revenue.
        segment_col: Column name containing segment labels.
        risk_band_col: Column name containing canonical risk bands ('high', 'medium', 'low').
        churn_window_days: Observation window in days (default 90).

    Returns:
        Dictionary containing aggregate metrics, risk band breakdown, segment breakdown,
        and cross-tabulation.
    """
    if df.empty:
        return {
            "n_customers": 0,
            "total_net_revenue": 0.0,
            "expected_revenue_at_risk": 0.0,
            "pct_revenue_at_risk": 0.0,
            "mean_churn_probability": 0.0,
            "risk_band_breakdown": {},
            "segment_breakdown": {},
            "cross_tabulation": {},
        }

    work_df = df.copy()
    work_df["_rev_at_risk"] = compute_customer_revenue_at_risk(
        work_df[prob_col],
        work_df[revenue_col],
        churn_window_days=churn_window_days,
    )

    n_cust = len(work_df)
    total_rev = float(work_df[revenue_col].sum())
    total_at_risk = float(work_df["_rev_at_risk"].sum())
    mean_prob = float(work_df[prob_col].mean())
    pct_at_risk = float((total_at_risk / total_rev) * 100.0) if total_rev > 0.0 else 0.0

    # 1. Risk band breakdown
    risk_band_breakdown: dict[str, dict[str, float | int]] = {}
    if risk_band_col in work_df.columns:
        for band in ["high", "medium", "low"]:
            b_df = work_df[work_df[risk_band_col] == band]
            b_count = len(b_df)
            b_rev = float(b_df[revenue_col].sum())
            b_risk = float(b_df["_rev_at_risk"].sum())
            b_prob = float(b_df[prob_col].mean()) if b_count > 0 else 0.0
            risk_band_breakdown[band] = {
                "n_customers": b_count,
                "pct_customers": round(b_count / n_cust * 100.0, 2) if n_cust > 0 else 0.0,
                "net_revenue": round(b_rev, 2),
                "expected_revenue_at_risk": round(b_risk, 2),
                "mean_churn_probability": round(b_prob, 4),
            }

    # 2. Segment breakdown
    segment_breakdown: dict[str, dict[str, float | int]] = {}
    if segment_col in work_df.columns:
        for seg_label, s_df in work_df.groupby(segment_col):
            s_count = len(s_df)
            s_rev = float(s_df[revenue_col].sum())
            s_risk = float(s_df["_rev_at_risk"].sum())
            s_prob = float(s_df[prob_col].mean()) if s_count > 0 else 0.0
            segment_breakdown[str(seg_label)] = {
                "n_customers": s_count,
                "pct_customers": round(s_count / n_cust * 100.0, 2) if n_cust > 0 else 0.0,
                "net_revenue": round(s_rev, 2),
                "expected_revenue_at_risk": round(s_risk, 2),
                "mean_churn_probability": round(s_prob, 4),
            }

    # 3. Cross-tabulation: Segments x Risk Bands
    cross_tab: dict[str, dict[str, dict[str, float | int]]] = {}
    if segment_col in work_df.columns and risk_band_col in work_df.columns:
        for seg_label in sorted(work_df[segment_col].unique()):
            cross_tab[str(seg_label)] = {}
            for band in ["high", "medium", "low"]:
                sub = work_df[
                    (work_df[segment_col] == seg_label) & (work_df[risk_band_col] == band)
                ]
                sub_count = len(sub)
                sub_risk = float(sub["_rev_at_risk"].sum())
                cross_tab[str(seg_label)][band] = {
                    "count": sub_count,
                    "expected_revenue_at_risk": round(sub_risk, 2),
                }

    return {
        "n_customers": n_cust,
        "total_net_revenue": round(total_rev, 2),
        "expected_revenue_at_risk": round(total_at_risk, 2),
        "pct_revenue_at_risk": round(pct_at_risk, 2),
        "mean_churn_probability": round(mean_prob, 4),
        "risk_band_breakdown": risk_band_breakdown,
        "segment_breakdown": segment_breakdown,
        "cross_tabulation": cross_tab,
    }
