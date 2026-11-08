"""RFM feature engineering, scoring, and rule-based segmentation.

Implements P3-01, TRD §5.2, PRD §7.3 (FR-CUS-01, FR-CUS-04), and DESIGN.md:
- Clock isolation: all calculations are strictly parameterized by an explicit `as_of` date.
- Population filtering: eligible customers have >= 1 non-return purchase within the
  trailing lookback window (`RP_RFM_WINDOW_DAYS`, default 730 days) up to `as_of`.
- Recency: `recency_days = (as_of - last_order_date).days` on non-return purchases.
- Frequency: distinct non-return purchase invoices in the lookback window.
- Monetary: net revenue (gross sales minus returns) clipped at minimum 0.0.
- Quintile scoring: R, F, M scored 1..5 with deterministic tie-breaking.
  R=5 is most recent, F=5 is most frequent, M=5 is highest spend.
- Rule-based segmentation: deterministic mapping to canonical business labels:
  Champions, Loyal, New / Promising, Big Spenders, At Risk, Hibernating.
- Clustering preprocessing: `log1p` followed by `StandardScaler` for K-Means.
- Pure functions supporting both in-memory DataFrames and database sessions.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pandas as pd
import sqlalchemy as sa
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

from retailpulse.db.models.core import FactSalesLine
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Canonical rule-based segment labels per TRD §5.2 and DESIGN.md §41
SEGMENT_CHAMPIONS = "Champions"
SEGMENT_LOYAL = "Loyal"
SEGMENT_NEW_PROMISING = "New / Promising"
SEGMENT_BIG_SPENDERS = "Big Spenders"
SEGMENT_AT_RISK = "At Risk"
SEGMENT_HIBERNATING = "Hibernating"

CANONICAL_SEGMENTS = (
    SEGMENT_CHAMPIONS,
    SEGMENT_LOYAL,
    SEGMENT_NEW_PROMISING,
    SEGMENT_BIG_SPENDERS,
    SEGMENT_AT_RISK,
    SEGMENT_HIBERNATING,
)


@dataclass(frozen=True)
class CustomerRFM:
    """Strongly-typed customer RFM metrics and segment assignment."""

    customer_id: int
    recency_days: int
    frequency: int
    monetary: float
    last_order_date: datetime.date
    first_order_date: datetime.date
    r_score: int
    f_score: int
    m_score: int
    rfm_score: str
    rule_segment: str


def assign_rule_based_segment(r_score: int, f_score: int, m_score: int) -> str:
    """Map RFM quintiles (1..5) to canonical business segments.

    Deterministic taxonomy per TRD §5.2 and DESIGN.md:
    1. Champions: r >= 4, f >= 4, m >= 4 (recent, frequent, highest spenders).
    2. Loyal: r >= 3, f >= 3 (consistent, regular shoppers).
    3. Big Spenders: m >= 4, r >= 3 (high lifetime spend, not dormant).
    4. New / Promising: r >= 4, f <= 2 (recently acquired, early lifecycle).
    5. At Risk: r <= 2, (f >= 3 or m >= 3) (past valuable shoppers now lapsing).
    6. Hibernating: all remaining customers (low recency, low engagement).

    Args:
        r_score: Recency score 1..5 (5 = most recent).
        f_score: Frequency score 1..5 (5 = most frequent).
        m_score: Monetary score 1..5 (5 = highest spend).

    Returns:
        Canonical segment string label.
    """
    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return SEGMENT_CHAMPIONS
    if r_score >= 3 and f_score >= 3:
        return SEGMENT_LOYAL
    if r_score >= 3 and m_score >= 4:
        return SEGMENT_BIG_SPENDERS
    if r_score >= 4 and f_score <= 2:
        return SEGMENT_NEW_PROMISING
    if r_score <= 2 and (f_score >= 3 or m_score >= 3):
        return SEGMENT_AT_RISK
    return SEGMENT_HIBERNATING


def _score_quintiles(
    df: pd.DataFrame,
    value_col: str,
    *,
    higher_is_better: bool = True,
    tie_breakers: list[tuple[str, bool]] | None = None,
) -> pd.Series:
    """Assign quintile scores 1..5 with deterministic ranking and tie-breaking.

    For frequency and monetary: `higher_is_better=True`, score 5 = highest.
    For recency_days: `higher_is_better=False`, score 5 = lowest days (most recent).

    Args:
        df: Input DataFrame containing `value_col` and tie-breaking columns.
        value_col: Column to bin into quintiles.
        higher_is_better: Whether higher values receive higher scores.
        tie_breakers: Secondary `(col_name, ascending_bool)` sort criteria.

    Returns:
        Integer Series with values in 1..5 indexed identical to `df.index`.
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=int, index=df.index)
    if n == 1:
        return pd.Series([3], index=df.index, dtype=int)

    sort_cols = [value_col]
    sort_ascending = [not higher_is_better if not higher_is_better else True]

    if tie_breakers:
        for t_col, t_asc in tie_breakers:
            if t_col in df.columns and t_col != value_col:
                sort_cols.append(t_col)
                sort_ascending.append(t_asc)

    if "customer_id" in df.columns and "customer_id" not in sort_cols:
        sort_cols.append("customer_id")
        sort_ascending.append(True)

    sorted_indices = df.sort_values(by=sort_cols, ascending=sort_ascending).index
    ranks = pd.Series(range(1, n + 1), index=sorted_indices)

    labels = [1, 2, 3, 4, 5] if higher_is_better else [5, 4, 3, 2, 1]
    scores = pd.qcut(ranks, q=5, labels=labels).astype(int)
    return scores.loc[df.index]


def _parse_order_date(data: pd.DataFrame) -> pd.Series:
    """Extract order date as datetime.date from date_key or invoice_ts."""
    if "date_key" in data.columns:
        first_val = data["date_key"].iloc[0]
        if isinstance(first_val, datetime.datetime):
            return data["date_key"].dt.date
        if isinstance(first_val, str):
            return pd.to_datetime(data["date_key"]).dt.date
        return data["date_key"]
    if "invoice_ts" in data.columns:
        return pd.to_datetime(data["invoice_ts"]).dt.date
    raise ValueError("DataFrame must contain 'date_key' or 'invoice_ts'")


def _derive_is_return(data: pd.DataFrame) -> pd.Series:
    """Infer boolean is_return flag from quantity or invoice number if absent."""
    if "is_return" in data.columns:
        return data["is_return"].astype(bool)
    if "quantity" in data.columns:
        return data["quantity"] < 0
    if "invoice_no" in data.columns:
        return data["invoice_no"].astype(str).str.upper().str.startswith("C")
    return pd.Series(False, index=data.index)


def _derive_line_amount(data: pd.DataFrame) -> pd.Series:
    """Derive numerical line amount from revenue, amount, or quantity * unit_price."""
    if "line_amount" in data.columns:
        return data["line_amount"].astype(float)
    if "quantity" in data.columns and "unit_price" in data.columns:
        return (data["quantity"] * data["unit_price"]).astype(float)
    if "revenue" in data.columns:
        return data["revenue"].astype(float)
    if "amount" in data.columns:
        return data["amount"].astype(float)
    raise ValueError("DataFrame must contain 'line_amount' or 'quantity' and 'unit_price'")


def _normalize_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names, datatypes, and derive required sales fields."""
    col_map = {
        "Customer ID": "customer_id",
        "CustomerID": "customer_id",
        "Invoice": "invoice_no",
        "InvoiceNo": "invoice_no",
        "InvoiceDate": "invoice_ts",
        "date": "date_key",
        "Quantity": "quantity",
        "UnitPrice": "unit_price",
        "Price": "unit_price",
    }
    data = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "customer_id" not in data.columns:
        raise ValueError("DataFrame must contain 'customer_id' column")

    data = data[data["customer_id"].notna()].copy()
    if data.empty:
        return data

    data["customer_id"] = data["customer_id"].astype(int)
    data["order_date"] = _parse_order_date(data)
    data["is_return"] = _derive_is_return(data)
    data["line_amount"] = _derive_line_amount(data)

    if "invoice_no" not in data.columns:
        data["invoice_no"] = data.index.astype(str)

    return data


def _filter_and_aggregate_rfm(
    data: pd.DataFrame,
    as_of: datetime.date,
    rfm_window_days: int,
) -> pd.DataFrame:
    """Filter window population and aggregate raw recency, frequency, and monetary metrics."""
    start_date = as_of - datetime.timedelta(days=rfm_window_days)
    in_window = (data["order_date"] >= start_date) & (data["order_date"] <= as_of)
    window_data = data[in_window].copy()
    if window_data.empty:
        return pd.DataFrame()

    # Eligible population: >= 1 non-return purchase (TRD §5.2)
    has_qty = "quantity" in window_data.columns
    is_purchase = (~window_data["is_return"]) & (window_data["quantity"] > 0 if has_qty else True)
    purchases = window_data[is_purchase].copy()
    eligible_ids = set(purchases["customer_id"].unique())

    if not eligible_ids:
        return pd.DataFrame()

    window_data = window_data[window_data["customer_id"].isin(eligible_ids)].copy()
    purchases = purchases[purchases["customer_id"].isin(eligible_ids)].copy()

    cust_purchases = (
        purchases.groupby("customer_id")
        .agg(
            last_order_date=("order_date", "max"),
            first_order_date=("order_date", "min"),
            frequency=("invoice_no", "nunique"),
        )
        .reset_index()
    )

    cust_revenue = (
        window_data.groupby("customer_id")["line_amount"]
        .sum()
        .reset_index()
        .rename(columns={"line_amount": "monetary"})
    )

    rfm = cust_purchases.merge(cust_revenue, on="customer_id", how="left")
    rfm["recency_days"] = rfm["last_order_date"].apply(lambda d: (as_of - d).days)
    rfm["monetary"] = rfm["monetary"].clip(lower=0.0).round(2)
    return rfm


def compute_rfm_from_df(
    df: pd.DataFrame,
    *,
    as_of: datetime.date,
    rfm_window_days: int = 730,
) -> pd.DataFrame:
    """Compute RFM metrics, quintiles, and rule segments from sales DataFrame.

    Implements FR-CUS-01, FR-CUS-04, and TRD §5.2.

    Args:
        df: Input sales transactions. Can be granular sales lines or invoice rows.
        as_of: Explicit reference cutoff date. Never reads system clock.
        rfm_window_days: Historical lookback window in days (default 730).

    Returns:
        DataFrame of customer RFM metrics and segment assignments.
    """
    empty_result = pd.DataFrame(
        columns=[
            "customer_id",
            "recency_days",
            "frequency",
            "monetary",
            "last_order_date",
            "first_order_date",
            "r_score",
            "f_score",
            "m_score",
            "rfm_score",
            "rule_segment",
        ]
    )
    if df.empty:
        return empty_result

    data = _normalize_sales_df(df)
    if data.empty:
        return empty_result

    rfm = _filter_and_aggregate_rfm(data, as_of=as_of, rfm_window_days=rfm_window_days)
    if rfm.empty:
        return empty_result

    rfm["r_score"] = _score_quintiles(
        rfm,
        "recency_days",
        higher_is_better=False,
        tie_breakers=[("monetary", False), ("frequency", False)],
    )

    rfm["f_score"] = _score_quintiles(
        rfm,
        "frequency",
        higher_is_better=True,
        tie_breakers=[("monetary", False), ("recency_days", True)],
    )

    rfm["m_score"] = _score_quintiles(
        rfm,
        "monetary",
        higher_is_better=True,
        tie_breakers=[("frequency", False), ("recency_days", True)],
    )

    rfm["rfm_score"] = (
        rfm["r_score"].astype(str) + rfm["f_score"].astype(str) + rfm["m_score"].astype(str)
    )

    rfm["rule_segment"] = [
        assign_rule_based_segment(r, f, m)
        for r, f, m in zip(rfm["r_score"], rfm["f_score"], rfm["m_score"], strict=True)
    ]

    cols = [
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "last_order_date",
        "first_order_date",
        "r_score",
        "f_score",
        "m_score",
        "rfm_score",
        "rule_segment",
    ]
    rfm = rfm[cols].sort_values("customer_id").reset_index(drop=True)

    logger.info(
        "computed rfm features",
        as_of=str(as_of),
        customer_count=len(rfm),
        rfm_window_days=rfm_window_days,
    )
    return rfm


def compute_rfm_from_db(
    session: Session,
    as_of: datetime.date,
    *,
    rfm_window_days: int = 730,
) -> pd.DataFrame:
    """Compute RFM features directly from the database core.fact_sales_line.

    Args:
        session: Active database session.
        as_of: Explicit reference cutoff date.
        rfm_window_days: Lookback window in calendar days (defaults to 730).

    Returns:
        DataFrame of customer RFM metrics and segment assignments.
    """
    start_date = as_of - datetime.timedelta(days=rfm_window_days)

    stmt = sa.select(
        FactSalesLine.customer_id,
        FactSalesLine.invoice_no,
        FactSalesLine.date_key,
        FactSalesLine.quantity,
        FactSalesLine.unit_price,
        FactSalesLine.line_amount,
        FactSalesLine.is_return,
    ).where(
        FactSalesLine.customer_id.isnot(None),
        FactSalesLine.date_key >= start_date,
        FactSalesLine.date_key <= as_of,
    )

    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "recency_days",
                "frequency",
                "monetary",
                "last_order_date",
                "first_order_date",
                "r_score",
                "f_score",
                "m_score",
                "rfm_score",
                "rule_segment",
            ]
        )

    df = pd.DataFrame([dict(r._mapping) for r in rows])
    return compute_rfm_from_df(df, as_of=as_of, rfm_window_days=rfm_window_days)


def transform_rfm_for_clustering(
    rfm_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[npt.NDArray[np.float64], StandardScaler]:
    """Transform RFM features for K-Means clustering per TRD §5.2.

    Applies `log1p(x)` transformation followed by `StandardScaler`.
    `log1p` handles the heavy skewness of recency, frequency, and monetary.

    Args:
        rfm_df: DataFrame containing `recency_days`, `frequency`, and `monetary`.
        scaler: Optional pre-fitted StandardScaler. If None, fits a new scaler.

    Returns:
        Tuple of (scaled feature array of shape (N, 3), fitted StandardScaler).
    """
    feature_cols = ["recency_days", "frequency", "monetary"]
    missing = [c for c in feature_cols if c not in rfm_df.columns]
    if missing:
        raise ValueError(f"Missing required RFM feature columns: {missing}")

    x_raw = rfm_df[feature_cols].to_numpy(dtype=float)
    x_log = np.log1p(np.maximum(x_raw, 0.0))

    if scaler is None:
        fitted_scaler = StandardScaler()
        x_scaled = fitted_scaler.fit_transform(x_log)
        return x_scaled, fitted_scaler

    x_scaled = scaler.transform(x_log)
    return x_scaled, scaler


def rfm_df_to_records(rfm_df: pd.DataFrame) -> list[CustomerRFM]:
    """Convert an RFM DataFrame into a list of strongly-typed CustomerRFM records."""
    records = []
    for _, row in rfm_df.iterrows():
        records.append(
            CustomerRFM(
                customer_id=int(row["customer_id"]),
                recency_days=int(row["recency_days"]),
                frequency=int(row["frequency"]),
                monetary=float(row["monetary"]),
                last_order_date=row["last_order_date"],
                first_order_date=row["first_order_date"],
                r_score=int(row["r_score"]),
                f_score=int(row["f_score"]),
                m_score=int(row["m_score"]),
                rfm_score=str(row["rfm_score"]),
                rule_segment=str(row["rule_segment"]),
            )
        )
    return records
