"""Customer churn label computation, monthly cutoffs, and purged temporal splits.

Implements P3-03, TRD §5.3, PRD §7.4 FR-CH-01, FR-CH-02 & TESTING:
- Monthly cutoffs from first date with >= 6 months of history through T + 90 <= max_date.
- Active customer population at T: at least 1 non-return purchase in (T - 365 days, T].
- Churn label: 1 if no non-return purchase in (T, T + RP_CHURN_WINDOW_DAYS], else 0.
- Purged temporal split: train, val, and test cutoffs strictly separated by the
  churn outcome window (90 days) to prevent label-to-feature leakage across splits.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.core import FactSalesLine
from retailpulse.features.churn_features import CHURN_FEATURE_COLUMNS, build_churn_features
from retailpulse.logging import get_logger

logger = get_logger(__name__)


def _advance_to_next_month(d: datetime.date) -> datetime.date:
    """Advance to the 1st day of the next month."""
    if d.month == 12:
        return datetime.date(d.year + 1, 1, 1)
    return datetime.date(d.year, d.month + 1, 1)


def generate_monthly_cutoffs(
    min_date: datetime.date,
    max_date: datetime.date,
    *,
    min_history_days: int = 180,
    churn_window_days: int = 90,
) -> list[datetime.date]:
    """Generate monthly cutoff dates on the 1st of each month.

    Starts from the first 1st-of-month date with at least `min_history_days`
    of history, ending at the last cutoff T where T + churn_window_days <= max_date.

    Args:
        min_date: Earliest available transaction date.
        max_date: Latest available transaction date.
        min_history_days: Minimum historical days required before first cutoff (default 180).
        churn_window_days: Outcome observation window length in days (default 90).

    Returns:
        List of cutoff dates chronologically sorted.
    """
    if min_date >= max_date:
        msg = f"min_date ({min_date}) must be strictly earlier than max_date ({max_date})"
        raise ValueError(msg)

    first_eligible = min_date + datetime.timedelta(days=min_history_days)
    current = first_eligible if first_eligible.day == 1 else _advance_to_next_month(first_eligible)

    latest_eligible = max_date - datetime.timedelta(days=churn_window_days)
    cutoffs: list[datetime.date] = []

    while current <= latest_eligible:
        cutoffs.append(current)
        current = _advance_to_next_month(current)

    return cutoffs


def create_purged_temporal_split(
    cutoffs: Sequence[datetime.date],
    *,
    churn_window_days: int = 90,
    val_size_cutoffs: int = 2,
    test_size_cutoffs: int = 1,
) -> tuple[list[datetime.date], list[datetime.date], list[datetime.date]]:
    """Partition monthly cutoffs into train, validation, and test blocks with purge gaps.

    Enforces that label windows never overlap subsequent split features:
      - Any training cutoff T_train must have:
          T_train + churn_window_days <= min(validation_cutoffs)
      - Any validation cutoff T_val must have:
          T_val + churn_window_days <= min(test_cutoffs)

    Args:
        cutoffs: Ordered sequence of cutoff dates.
        churn_window_days: Outcome window in days (default 90).
        val_size_cutoffs: Number of cutoffs in the validation block (default 2).
        test_size_cutoffs: Number of cutoffs in the test block (default 1).

    Returns:
        Tuple of (train_cutoffs, validation_cutoffs, test_cutoffs).

    Raises:
        ValueError: If there are insufficient cutoffs to satisfy purge gaps.
    """
    sorted_cutoffs = sorted(cutoffs)
    min_required = test_size_cutoffs + val_size_cutoffs + 1
    if len(sorted_cutoffs) < min_required:
        raise ValueError(
            f"Insufficient cutoffs: got {len(sorted_cutoffs)}, need at least {min_required} "
            f"for test={test_size_cutoffs}, val={val_size_cutoffs}, train >= 1"
        )

    test_cutoffs = list(sorted_cutoffs[-test_size_cutoffs:])
    test_first = test_cutoffs[0]

    # Validation cutoffs: latest available cutoffs whose outcome window ends before test_first
    val_candidates = [
        c
        for c in sorted_cutoffs[:-test_size_cutoffs]
        if c + datetime.timedelta(days=churn_window_days) <= test_first
    ]
    if len(val_candidates) < val_size_cutoffs:
        raise ValueError(
            f"Cannot allocate {val_size_cutoffs} validation cutoffs with purge gap of "
            f"{churn_window_days} days before test start {test_first}. "
            f"Only {len(val_candidates)} candidates available."
        )

    val_cutoffs = val_candidates[-val_size_cutoffs:]
    val_first = val_cutoffs[0]

    # Training cutoffs: cutoffs whose outcome window ends before val_first
    train_cutoffs = [
        c for c in sorted_cutoffs if c + datetime.timedelta(days=churn_window_days) <= val_first
    ]
    if not train_cutoffs:
        raise ValueError(
            f"Cannot allocate training cutoffs with purge gap of {churn_window_days} days "
            f"before validation start {val_first}."
        )

    # Sanity checks: assert zero leakage across boundaries
    max_train = max(train_cutoffs)
    if max_train + datetime.timedelta(days=churn_window_days) > val_first:
        raise AssertionError("Train cutoff label window leaks into validation features")

    max_val = max(val_cutoffs)
    if max_val + datetime.timedelta(days=churn_window_days) > test_first:
        raise AssertionError("Validation cutoff label window leaks into test features")

    return train_cutoffs, val_cutoffs, test_cutoffs


def compute_churn_labels(
    transactions_df: pd.DataFrame,
    cutoff: datetime.date,
    *,
    churn_window_days: int = 90,
    active_lookback_days: int = 365,
) -> pd.DataFrame:
    """Compute churn labels for active customers at a cutoff date T.

    Active population at T:
      - Has >= 1 purchase with quantity > 0 and unit_price > 0 in (T - active_lookback_days, T].
    Churn label:
      - 1 if the customer has 0 non-return purchases in (T, T + churn_window_days].
      - 0 if the customer made >= 1 non-return purchase in that window.

    Args:
        transactions_df: DataFrame with transaction rows. Must contain:
            - `customer_id`
            - `date_key` or `invoice_date` or `invoice_ts`
            - `quantity`
            - `unit_price`
        cutoff: Cutoff date T.
        churn_window_days: Churn label observation window length (default 90).
        active_lookback_days: Trailing window to define active customer population (default 365).

    Returns:
        DataFrame with columns:
            - `customer_id`: int
            - `cutoff_date`: datetime.date
            - `churn_label`: int (1 for churned, 0 for retained)
            - `days_to_next_purchase`: float (NaN if churned, else days from T to first purchase)
    """
    if transactions_df.empty:
        return pd.DataFrame(
            columns=["customer_id", "cutoff_date", "churn_label", "days_to_next_purchase"]
        )

    df = transactions_df.copy()

    # Normalize date column
    if "invoice_date" in df.columns:
        date_series = pd.to_datetime(df["invoice_date"]).dt.date
    elif "date_key" in df.columns:
        date_series = pd.to_datetime(df["date_key"]).dt.date
    elif "invoice_ts" in df.columns:
        date_series = pd.to_datetime(df["invoice_ts"]).dt.date
    else:
        raise ValueError("transactions_df must contain 'invoice_date', 'date_key', or 'invoice_ts'")

    df["_date"] = date_series

    # Identify non-return (positive purchase) transactions
    pos_mask = (df["quantity"] > 0) & (df["unit_price"] > 0) & (df["customer_id"].notna())
    pos_df = df[pos_mask]

    active_start = cutoff - datetime.timedelta(days=active_lookback_days)
    outcome_end = cutoff + datetime.timedelta(days=churn_window_days)

    # 1. Active customer population in (cutoff - active_lookback_days, cutoff]
    active_mask = (pos_df["_date"] > active_start) & (pos_df["_date"] <= cutoff)
    active_customers = pos_df[active_mask]["customer_id"].unique()

    if len(active_customers) == 0:
        return pd.DataFrame(
            columns=["customer_id", "cutoff_date", "churn_label", "days_to_next_purchase"]
        )

    # 2. Future purchases in (cutoff, cutoff + churn_window_days]
    future_mask = (pos_df["_date"] > cutoff) & (pos_df["_date"] <= outcome_end)
    future_df = pos_df[future_mask]

    # Calculate earliest future purchase date per customer
    future_first = future_df.groupby("customer_id")["_date"].min()

    # 3. Assemble labels
    records: list[dict[str, object]] = []
    for cid in active_customers:
        cid_int = int(cid)
        if cid in future_first.index:
            first_future_date = future_first[cid]
            days_to_next = float((first_future_date - cutoff).days)
            label = 0
        else:
            days_to_next = np.nan
            label = 1

        records.append(
            {
                "customer_id": cid_int,
                "cutoff_date": cutoff,
                "churn_label": label,
                "days_to_next_purchase": days_to_next,
            }
        )

    result_df = pd.DataFrame(records).sort_values("customer_id").reset_index(drop=True)
    result_df["customer_id"] = result_df["customer_id"].astype(np.int64)
    result_df["churn_label"] = result_df["churn_label"].astype(np.int64)
    result_df["days_to_next_purchase"] = result_df["days_to_next_purchase"].astype(float)
    return result_df


def compute_churn_labels_from_db(
    session: Session,
    cutoff: datetime.date,
    *,
    churn_window_days: int = 90,
    active_lookback_days: int = 365,
) -> pd.DataFrame:
    """Compute churn labels directly from database FactSalesLine table.

    Args:
        session: SQLAlchemy session.
        cutoff: Cutoff date T.
        churn_window_days: Churn label observation window length (default 90).
        active_lookback_days: Trailing window to define active customers (default 365).

    Returns:
        DataFrame with columns:
            `customer_id`, `cutoff_date`, `churn_label`, `days_to_next_purchase`.
    """
    start_date = cutoff - datetime.timedelta(days=active_lookback_days)
    end_date = cutoff + datetime.timedelta(days=churn_window_days)

    stmt = sa.select(
        FactSalesLine.customer_id,
        FactSalesLine.date_key,
        FactSalesLine.quantity,
        FactSalesLine.unit_price,
        FactSalesLine.invoice_no,
    ).where(
        FactSalesLine.date_key > start_date,
        FactSalesLine.date_key <= end_date,
        FactSalesLine.customer_id.isnot(None),
    )

    rows = session.execute(stmt).all()
    if not rows:
        return pd.DataFrame(
            columns=["customer_id", "cutoff_date", "churn_label", "days_to_next_purchase"]
        )

    df = pd.DataFrame(
        rows,
        columns=["customer_id", "date_key", "quantity", "unit_price", "invoice_no"],
    )
    return compute_churn_labels(
        df,
        cutoff,
        churn_window_days=churn_window_days,
        active_lookback_days=active_lookback_days,
    )


def build_churn_dataset(
    transactions_df: pd.DataFrame,
    cutoffs: Sequence[datetime.date],
    *,
    churn_window_days: int = 90,
    lookback_days: int = 365,
) -> pd.DataFrame:
    """Build multi-cutoff modeling dataset merging features and future labels.

    For each cutoff T:
      - Computes features using data <= T via `build_churn_features`.
      - Computes churn labels using positive sales in (T, T + churn_window_days].
      - Merges on `customer_id` and appends to the unified dataset.

    Args:
        transactions_df: Raw or cleaned transactions.
        cutoffs: Ordered sequence of cutoff dates.
        churn_window_days: Churn outcome window (default 90).
        lookback_days: Active customer trailing window (default 365).

    Returns:
        Unified DataFrame with features, `cutoff_date`, and `churn_label`.
    """
    dfs: list[pd.DataFrame] = []

    for cutoff in cutoffs:
        feat_df = build_churn_features(
            transactions_df,
            cutoff,
            lookback_days=lookback_days,
        )
        if feat_df.empty:
            continue

        label_df = compute_churn_labels(
            transactions_df,
            cutoff,
            churn_window_days=churn_window_days,
            active_lookback_days=lookback_days,
        )

        merged = feat_df.merge(
            label_df[["customer_id", "churn_label", "days_to_next_purchase"]],
            on="customer_id",
            how="inner",
        )
        dfs.append(merged)

    if not dfs:
        cols = [
            "customer_id",
            "cutoff_date",
            *CHURN_FEATURE_COLUMNS,
            "churn_label",
            "days_to_next_purchase",
        ]
        return pd.DataFrame(columns=cols)

    return pd.concat(dfs, ignore_index=True)
