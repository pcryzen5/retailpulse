"""Customer churn feature engineering and dataset generation.

Implements P3-03, TRD §5.3, PRD §7.4 FR-CH-01, FR-CH-02:
- Strict temporal leakage guard: transactions with date > cutoff are strictly excluded.
- Features computed for active customers with >= 1 non-return purchase in trailing 365 days.
- Rich behavioral feature set:
    - recency, tenure
    - orders across 30, 90, 180, 365 days
    - net revenue across 90 and 365 days
    - order count trend (recent 90 vs prior 90)
    - avg order value, items per order, distinct SKUs, return rate
    - inter-purchase gap mean and std, overdue ratio (recency / gap_mean)
    - single order indicator, country, month of cutoff
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.core import FactSalesLine
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Feature column ordering for modeling
CHURN_FEATURE_COLUMNS = [
    "recency_days",
    "tenure_days",
    "orders_30d",
    "orders_90d",
    "orders_180d",
    "orders_365d",
    "orders_prior_90d",
    "order_count_trend",
    "net_revenue_90d",
    "net_revenue_365d",
    "avg_order_value",
    "items_per_order",
    "distinct_skus",
    "return_rate",
    "gap_mean",
    "gap_std",
    "overdue_ratio",
    "has_single_order",
    "country",
    "month_of_cutoff",
]


def _normalize_date_column(df: pd.DataFrame) -> pd.Series:
    """Normalize date column from various possible column names."""
    if "invoice_date" in df.columns:
        return pd.to_datetime(df["invoice_date"]).dt.date
    if "date_key" in df.columns:
        return pd.to_datetime(df["date_key"]).dt.date
    if "invoice_ts" in df.columns:
        return pd.to_datetime(df["invoice_ts"]).dt.date
    raise ValueError("DataFrame must contain 'invoice_date', 'date_key', or 'invoice_ts'")


def build_churn_features(  # noqa: PLR0915
    transactions_df: pd.DataFrame,
    cutoff: datetime.date,
    *,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Extract customer churn features as of cutoff T using strictly data <= T.

    Active customer population at T:
      - Has >= 1 purchase with quantity > 0 and unit_price > 0 in (T - lookback_days, T].

    Args:
        transactions_df: DataFrame of raw sales lines or fact sales records.
        cutoff: Cutoff date T. Any record with date > T is strictly filtered out.
        lookback_days: Active customer lookback window in days (default 365).

    Returns:
        DataFrame with index `customer_id` and all churn feature columns.
    """
    if transactions_df.empty:
        return pd.DataFrame(columns=["customer_id", "cutoff_date", *CHURN_FEATURE_COLUMNS])

    df = transactions_df.copy()
    df["_date"] = _normalize_date_column(df)

    # 1. STRICT LEAKAGE GUARD: Only consider transactions up to cutoff T
    hist_mask = (df["_date"] <= cutoff) & (df["customer_id"].notna())
    hist = df[hist_mask].copy()
    if hist.empty:
        return pd.DataFrame(columns=["customer_id", "cutoff_date", *CHURN_FEATURE_COLUMNS])

    # Compute line amount if not present
    if "line_amount" not in hist.columns:
        hist["line_amount"] = hist["quantity"] * hist["unit_price"]

    # Normalize customer_id to int64
    hist["customer_id"] = hist["customer_id"].astype(np.int64)

    # Positive sales lines (non-returns)
    pos_hist = hist[(hist["quantity"] > 0) & (hist["unit_price"] > 0)].copy()

    # Active population in (cutoff - lookback_days, cutoff]
    w365_start = cutoff - datetime.timedelta(days=lookback_days)
    pos_365 = pos_hist[pos_hist["_date"] > w365_start]
    active_cids = pos_365["customer_id"].unique()

    if len(active_cids) == 0:
        return pd.DataFrame(columns=["customer_id", "cutoff_date", *CHURN_FEATURE_COLUMNS])

    # Windows relative to cutoff T
    w180_start = cutoff - datetime.timedelta(days=180)
    w90_start = cutoff - datetime.timedelta(days=90)
    w30_start = cutoff - datetime.timedelta(days=30)

    # 2. Tenure & Recency
    # Tenure: days between first known purchase in history and cutoff T
    tenure_s = pos_hist.groupby("customer_id")["_date"].min().apply(lambda d: (cutoff - d).days)

    # Recency: days between latest purchase in 365d and cutoff T
    recency_s = pos_365.groupby("customer_id")["_date"].max().apply(lambda d: (cutoff - d).days)

    # 3. Order counts across sub-windows
    ord_30 = pos_365[pos_365["_date"] > w30_start].groupby("customer_id")["invoice_no"].nunique()
    ord_90 = pos_365[pos_365["_date"] > w90_start].groupby("customer_id")["invoice_no"].nunique()
    ord_180 = pos_365[pos_365["_date"] > w180_start].groupby("customer_id")["invoice_no"].nunique()
    ord_365 = pos_365.groupby("customer_id")["invoice_no"].nunique()

    # Prior 90 days: (cutoff - 180, cutoff - 90]
    ord_prior90 = (
        pos_365[(pos_365["_date"] > w180_start) & (pos_365["_date"] <= w90_start)]
        .groupby("customer_id")["invoice_no"]
        .nunique()
    )

    # 4. Net revenue in 90d and 365d (all transactions including returns)
    hist_365 = hist[hist["_date"] > w365_start]
    hist_90 = hist_365[hist_365["_date"] > w90_start]
    rev_90 = hist_90.groupby("customer_id")["line_amount"].sum()
    rev_365 = hist_365.groupby("customer_id")["line_amount"].sum()

    # 5. Quantities and SKU counts in 365d
    if "sku" in pos_365.columns:
        sku_col = "sku"
    elif "product_id" in pos_365.columns:
        sku_col = "product_id"
    else:
        sku_col = "description"
    sku_365 = pos_365.groupby("customer_id")[sku_col].nunique()
    pos_qty_365 = pos_365.groupby("customer_id")["quantity"].sum()

    # Returns in 365d (lines with quantity < 0)
    ret_365 = hist_365[hist_365["quantity"] < 0]
    ret_qty_365 = ret_365.groupby("customer_id")["quantity"].sum().abs()

    # 6. Inter-purchase gaps in history up to cutoff T
    dates_df = (
        pos_hist[["customer_id", "_date"]].drop_duplicates().sort_values(["customer_id", "_date"])
    )
    dates_df["_prev_date"] = dates_df.groupby("customer_id")["_date"].shift(1)
    dates_df["_gap"] = (
        pd.to_datetime(dates_df["_date"]) - pd.to_datetime(dates_df["_prev_date"])
    ).dt.days

    gap_valid = dates_df.dropna(subset=["_gap"])
    gap_mean = gap_valid.groupby("customer_id")["_gap"].mean()
    # std with ddof=1 produces NaN when only 1 gap; replace with 0.0 if exactly 1 gap
    gap_count = gap_valid.groupby("customer_id")["_gap"].count()
    gap_std_raw = gap_valid.groupby("customer_id")["_gap"].std()
    gap_std = gap_std_raw.where(gap_count > 1, 0.0)

    # 7. Customer country (mode or last country recorded)
    if "country" in hist.columns:
        cust_country = hist.groupby("customer_id")["country"].last()
    else:
        cust_country = pd.Series("United Kingdom", index=active_cids)

    # 8. Assemble feature dataframe
    f_df = (
        pd.DataFrame({"customer_id": active_cids}).sort_values("customer_id").reset_index(drop=True)
    )
    f_df["cutoff_date"] = cutoff

    f_df["recency_days"] = f_df["customer_id"].map(recency_s).astype(float)
    f_df["tenure_days"] = f_df["customer_id"].map(tenure_s).astype(float)

    f_df["orders_30d"] = f_df["customer_id"].map(ord_30).fillna(0).astype(np.int64)
    f_df["orders_90d"] = f_df["customer_id"].map(ord_90).fillna(0).astype(np.int64)
    f_df["orders_180d"] = f_df["customer_id"].map(ord_180).fillna(0).astype(np.int64)
    f_df["orders_365d"] = f_df["customer_id"].map(ord_365).fillna(0).astype(np.int64)
    f_df["orders_prior_90d"] = f_df["customer_id"].map(ord_prior90).fillna(0).astype(np.int64)

    # Trend: normalized diff between recent 90d and prior 90d orders
    f_df["order_count_trend"] = (
        (f_df["orders_90d"] - f_df["orders_prior_90d"])
        / np.maximum(1.0, f_df["orders_90d"] + f_df["orders_prior_90d"])
    ).astype(float)

    f_df["net_revenue_90d"] = f_df["customer_id"].map(rev_90).fillna(0.0).astype(float)
    f_df["net_revenue_365d"] = f_df["customer_id"].map(rev_365).fillna(0.0).astype(float)

    f_df["avg_order_value"] = (
        f_df["net_revenue_365d"] / np.maximum(1, f_df["orders_365d"])
    ).astype(float)

    total_units_pos = f_df["customer_id"].map(pos_qty_365).fillna(0.0).astype(float)
    f_df["items_per_order"] = (total_units_pos / np.maximum(1, f_df["orders_365d"])).astype(float)
    f_df["distinct_skus"] = f_df["customer_id"].map(sku_365).fillna(0).astype(np.int64)

    ret_units = f_df["customer_id"].map(ret_qty_365).fillna(0.0).astype(float)
    f_df["return_rate"] = (ret_units / np.maximum(1.0, total_units_pos + ret_units)).astype(float)

    f_df["gap_mean"] = f_df["customer_id"].map(gap_mean).astype(float)
    f_df["gap_std"] = f_df["customer_id"].map(gap_std).astype(float)

    # Overdue ratio: recency / gap_mean (NaN if gap_mean is NaN or 0)
    gap_mean_val = f_df["gap_mean"]
    f_df["overdue_ratio"] = np.where(
        gap_mean_val > 0,
        f_df["recency_days"] / gap_mean_val,
        np.nan,
    )

    f_df["has_single_order"] = (f_df["orders_365d"] == 1).astype(np.int64)
    f_df["country"] = f_df["customer_id"].map(cust_country).fillna("United Kingdom").astype(str)
    f_df["month_of_cutoff"] = int(cutoff.month)

    # Return with standardized column order
    cols = ["customer_id", "cutoff_date", *CHURN_FEATURE_COLUMNS]
    return f_df[cols]


def build_churn_features_from_db(
    session: Session,
    cutoff: datetime.date,
    *,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Extract churn features directly from database FactSalesLine table.

    Args:
        session: SQLAlchemy session.
        cutoff: Cutoff date T.
        lookback_days: Trailing window to define active customers (default 365).

    Returns:
        DataFrame with churn features for active customers as of cutoff T.
    """
    stmt = sa.select(
        FactSalesLine.customer_id,
        FactSalesLine.date_key,
        FactSalesLine.quantity,
        FactSalesLine.unit_price,
        FactSalesLine.invoice_no,
        FactSalesLine.product_id,
    ).where(
        FactSalesLine.date_key <= cutoff,
        FactSalesLine.customer_id.isnot(None),
    )

    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(columns=["customer_id", "cutoff_date", *CHURN_FEATURE_COLUMNS])

    df = pd.DataFrame(
        rows,
        columns=["customer_id", "date_key", "quantity", "unit_price", "invoice_no", "product_id"],
    )
    return build_churn_features(df, cutoff, lookback_days=lookback_days)
