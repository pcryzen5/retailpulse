"""Feature engineering and series construction for demand forecasting.

Implements P2-01 & TRD §5.1:
- Top-N series selection by trailing-365-day revenue up to as_of
- Product eligibility assessment (first sale >= 90 days ago, >= 20 non-zero sales days)
- Zero-filled trading-day series generation from sparse fact_daily_sales
- Target clipping for wholesale spikes at 99.5th percentile per series
- Leakage-safe feature extraction (lags, rolling stats, calendar, direct strategy horizon)
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.core import DimDate, DimProduct, FactDailySales
from retailpulse.etl.calendar import (
    days_to_christmas,
    default_christmas_closure_windows,
    is_in_closure_window,
)
from retailpulse.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ProductEligibility:
    """Eligibility status of a product for statistical and ML forecasting."""

    product_id: int | str
    sku: str
    is_eligible: bool
    ineligibility_reason: str | None
    first_sale_date: datetime.date | None
    sales_days_count: int
    trailing_revenue: float


def select_top_products(
    session: Session,
    as_of: datetime.date,
    *,
    top_n: int = 300,
    trailing_days: int = 365,
) -> list[DimProduct]:
    """Select top-N product entities by trailing revenue up to as_of.

    Args:
        session: Active database session.
        as_of: Reference cutoff date (data_as_of).
        top_n: Maximum number of products to return.
        trailing_days: Lookback window in calendar days (defaults to 365).

    Returns:
        List of DimProduct entities sorted descending by trailing revenue.
    """
    start_date = as_of - datetime.timedelta(days=trailing_days - 1)
    stmt = (
        sa.select(
            DimProduct,
            sa.func.coalesce(sa.func.sum(FactDailySales.revenue), 0).label("tot_rev"),
        )
        .join(FactDailySales, DimProduct.product_id == FactDailySales.product_id)
        .where(
            FactDailySales.date_key >= start_date,
            FactDailySales.date_key <= as_of,
            DimProduct.is_product.is_(True),
        )
        .group_by(DimProduct.product_id)
        .order_by(sa.desc("tot_rev"))
        .limit(top_n)
    )
    results = session.execute(stmt).scalars().all()
    logger.info("selected top products for forecasting", count=len(results), as_of=str(as_of))
    return list(results)


def select_top_products_df(
    daily_sales_df: pd.DataFrame,
    product_meta_df: pd.DataFrame,
    as_of: datetime.date,
    *,
    top_n: int = 300,
    trailing_days: int = 365,
) -> list[int | str]:
    """In-memory top-N product ID selection from daily sales DataFrame.

    Args:
        daily_sales_df: DataFrame with product_id, date_key, revenue.
        product_meta_df: DataFrame with product_id, sku, is_product.
        as_of: Cutoff date.
        top_n: Number of products to select.
        trailing_days: Lookback window in days.

    Returns:
        List of selected product IDs ordered descending by revenue.
    """
    start_date = as_of - datetime.timedelta(days=trailing_days - 1)

    # Filter to real products
    prod_mask = product_meta_df["is_product"].fillna(True).astype(bool)
    valid_ids = set(product_meta_df.loc[prod_mask, "product_id"])

    # Filter sales window
    mask = (
        (daily_sales_df["date_key"] >= start_date)
        & (daily_sales_df["date_key"] <= as_of)
        & (daily_sales_df["product_id"].isin(valid_ids))
    )
    filtered = daily_sales_df.loc[mask]

    grouped = (
        filtered.groupby("product_id")["revenue"].sum().sort_values(ascending=False).head(top_n)
    )
    return list(grouped.index)


def evaluate_product_eligibility(
    session: Session,
    as_of: datetime.date,
    product_ids: Sequence[int] | None = None,
    *,
    min_history_days: int = 90,
    min_sales_days: int = 20,
) -> dict[int, ProductEligibility]:
    """Assess forecasting eligibility for product IDs via database queries.

    Per TRD §5.1, eligible if:
    - first_sale_date <= as_of - min_history_days (>= 90 days of history)
    - >= 20 non-zero sales days in the trailing 365 days
    - is_product is True
    """
    trailing_start = as_of - datetime.timedelta(days=364)

    prod_stmt = sa.select(DimProduct)
    if product_ids is not None:
        prod_stmt = prod_stmt.where(DimProduct.product_id.in_(product_ids))
    products = session.execute(prod_stmt).scalars().all()

    # Query trailing non-zero sales days and revenue
    sales_stmt = (
        sa.select(
            FactDailySales.product_id,
            sa.func.count(FactDailySales.date_key).label("sales_days"),
            sa.func.coalesce(sa.func.sum(FactDailySales.revenue), 0).label("tot_rev"),
        )
        .where(
            FactDailySales.date_key >= trailing_start,
            FactDailySales.date_key <= as_of,
            FactDailySales.units_sold > 0,
        )
        .group_by(FactDailySales.product_id)
    )
    sales_rows = session.execute(sales_stmt).all()
    sales_map = {r.product_id: (int(r.sales_days), float(r.tot_rev)) for r in sales_rows}

    eligibility: dict[int, ProductEligibility] = {}
    for p in products:
        sales_days, tot_rev = sales_map.get(p.product_id, (0, 0.0))
        is_eligible = True
        reason: str | None = None

        if not p.is_product:
            is_eligible = False
            reason = "not_product"
        elif p.first_sale_date is None or (as_of - p.first_sale_date).days < min_history_days:
            is_eligible = False
            reason = "history_too_short"
        elif sales_days < min_sales_days:
            is_eligible = False
            reason = "insufficient_sales_days"

        eligibility[p.product_id] = ProductEligibility(
            product_id=p.product_id,
            sku=p.sku,
            is_eligible=is_eligible,
            ineligibility_reason=reason,
            first_sale_date=p.first_sale_date,
            sales_days_count=sales_days,
            trailing_revenue=tot_rev,
        )

    return eligibility


def evaluate_product_eligibility_df(
    daily_sales_df: pd.DataFrame,
    product_meta_df: pd.DataFrame,
    as_of: datetime.date,
    *,
    min_history_days: int = 90,
    min_sales_days: int = 20,
) -> dict[Any, ProductEligibility]:
    """In-memory eligibility assessment using Pandas DataFrames."""
    trailing_start = as_of - datetime.timedelta(days=364)

    # Trailing active sales filter
    mask = (
        (daily_sales_df["date_key"] >= trailing_start)
        & (daily_sales_df["date_key"] <= as_of)
        & (daily_sales_df["units_sold"] > 0)
    )
    active_sales = daily_sales_df.loc[mask]

    grouped = (
        active_sales.groupby("product_id")
        .agg(
            sales_days=("date_key", "nunique"),
            tot_rev=(
                ("revenue", "sum") if "revenue" in active_sales.columns else ("units_sold", "count")
            ),
        )
        .reset_index()
    )
    sales_map = {
        row["product_id"]: (int(row["sales_days"]), float(row["tot_rev"]))
        for _, row in grouped.iterrows()
    }

    eligibility: dict[Any, ProductEligibility] = {}
    for _, row in product_meta_df.iterrows():
        pid = row["product_id"]
        sku = str(row.get("sku", pid))
        is_product = bool(row.get("is_product", True))
        first_sale = row.get("first_sale_date")
        if pd.isna(first_sale):
            first_sale = None

        sales_days, tot_rev = sales_map.get(pid, (0, 0.0))
        is_eligible = True
        reason: str | None = None

        if not is_product:
            is_eligible = False
            reason = "not_product"
        elif first_sale is None or (as_of - first_sale).days < min_history_days:
            is_eligible = False
            reason = "history_too_short"
        elif sales_days < min_sales_days:
            is_eligible = False
            reason = "insufficient_sales_days"

        eligibility[pid] = ProductEligibility(
            product_id=pid,
            sku=sku,
            is_eligible=is_eligible,
            ineligibility_reason=reason,
            first_sale_date=first_sale,
            sales_days_count=sales_days,
            trailing_revenue=tot_rev,
        )

    return eligibility


def _build_single_product_series(
    pid: int | str,
    p_sales: pd.DataFrame,
    trading_dates: Sequence[datetime.date],
    holiday_map: dict[datetime.date, bool],
    abc: str,
) -> pd.DataFrame:
    """Build continuous series for a single product over trading dates."""
    records = []
    for d in trading_dates:
        units = 0
        price: float | None = None
        if d in p_sales.index:
            row = p_sales.loc[d]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            units = int(row.get("units_sold", 0))
            raw_price = row.get("avg_unit_price", row.get("price"))
            if not pd.isna(raw_price) and raw_price is not None:
                price = float(raw_price)

        records.append({"date": d, "units_sold": units, "price": price})

    prod_df = pd.DataFrame(records)

    # Forward-fill price, back-fill median price
    if prod_df["price"].notna().any():
        median_p = float(prod_df["price"].dropna().median())
        prod_df["price"] = prod_df["price"].ffill().fillna(median_p)
    else:
        prod_df["price"] = 0.0

    prod_df["series_id"] = pid
    prod_df["product_id"] = pid
    prod_df["y"] = prod_df["units_sold"]
    prod_df["is_holiday"] = [holiday_map.get(d, False) for d in prod_df["date"]]
    prod_df["abc_class"] = abc

    return prod_df[["series_id", "product_id", "date", "y", "price", "is_holiday", "abc_class"]]


def build_zero_filled_series_df(
    daily_sales_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    product_meta_df: pd.DataFrame,
    product_ids: Sequence[int | str],
    *,
    start_date: datetime.date,
    as_of: datetime.date,
) -> pd.DataFrame:
    """Build continuous, zero-filled daily sales series aligned with trading days."""
    cal_mask = (
        (calendar_df["date_key"] >= start_date)
        & (calendar_df["date_key"] <= as_of)
        & (calendar_df["is_trading_day"].fillna(True).astype(bool))
    )
    trading_cal = calendar_df.loc[cal_mask].sort_values("date_key")
    trading_dates = list(trading_cal["date_key"])

    if not trading_dates:
        return pd.DataFrame(
            columns=["series_id", "product_id", "date", "y", "price", "is_holiday", "abc_class"]
        )

    holiday_map = dict(
        zip(
            trading_cal["date_key"],
            trading_cal["is_holiday"].fillna(False).astype(bool),
            strict=False,
        )
    )
    meta_indexed = product_meta_df.set_index("product_id")
    series_dfs: list[pd.DataFrame] = []

    for pid in product_ids:
        p_sales = daily_sales_df[daily_sales_df["product_id"] == pid].set_index("date_key")
        meta_row = meta_indexed.loc[pid] if pid in meta_indexed.index else pd.Series()
        abc = str(meta_row.get("abc_class", "C")) if not pd.isna(meta_row.get("abc_class")) else "C"

        single_df = _build_single_product_series(
            pid=pid,
            p_sales=p_sales,
            trading_dates=trading_dates,
            holiday_map=holiday_map,
            abc=abc,
        )
        series_dfs.append(single_df)

    if not series_dfs:
        return pd.DataFrame(
            columns=["series_id", "product_id", "date", "y", "price", "is_holiday", "abc_class"]
        )

    return pd.concat(series_dfs, ignore_index=True)


def build_zero_filled_series(
    session: Session,
    product_ids: Sequence[int],
    start_date: datetime.date,
    as_of: datetime.date,
    *,
    store_id: int = 1,
) -> pd.DataFrame:
    """Build zero-filled trading-day series directly from database tables."""
    cal_stmt = sa.select(DimDate.date_key, DimDate.is_trading_day, DimDate.is_holiday).where(
        DimDate.date_key >= start_date, DimDate.date_key <= as_of
    )
    cal_rows = session.execute(cal_stmt).all()
    cal_df = pd.DataFrame([dict(r._mapping) for r in cal_rows])

    sales_stmt = sa.select(
        FactDailySales.product_id,
        FactDailySales.date_key,
        FactDailySales.units_sold,
        FactDailySales.avg_unit_price,
    ).where(
        FactDailySales.product_id.in_(product_ids),
        FactDailySales.store_id == store_id,
        FactDailySales.date_key >= start_date,
        FactDailySales.date_key <= as_of,
    )
    sales_rows = session.execute(sales_stmt).all()
    sales_df = pd.DataFrame([dict(r._mapping) for r in sales_rows])
    if sales_df.empty:
        sales_df = pd.DataFrame(columns=["product_id", "date_key", "units_sold", "avg_unit_price"])

    prod_stmt = sa.select(DimProduct.product_id, DimProduct.sku, DimProduct.abc_class).where(
        DimProduct.product_id.in_(product_ids)
    )
    prod_rows = session.execute(prod_stmt).all()
    prod_df = pd.DataFrame([dict(r._mapping) for r in prod_rows])

    return build_zero_filled_series_df(
        daily_sales_df=sales_df,
        calendar_df=cal_df,
        product_meta_df=prod_df,
        product_ids=product_ids,
        start_date=start_date,
        as_of=as_of,
    )


def clip_training_targets(
    df: pd.DataFrame,
    *,
    target_col: str = "y",
    group_col: str = "series_id",
    quantile: float = 0.995,
) -> pd.DataFrame:
    """Clip target values per series at the given percentile to prevent wholesale order distortion.

    Per TRD §5.1:
    - Training targets clipped at 99.5th percentile per series.
    - Evaluation always uses unclipped actuals.
    """
    out = df.copy()

    def _clip_series(group: pd.DataFrame) -> pd.DataFrame:
        cap = float(group[target_col].quantile(quantile))
        group[target_col] = group[target_col].clip(upper=cap)
        return group

    return out.groupby(group_col, group_keys=False).apply(_clip_series)


def _compute_origin_series_stats(
    group_sorted: pd.DataFrame,
    origin: datetime.date,
) -> dict[str, Any]:
    """Compute lag and rolling summary statistics strictly for history <= origin."""
    n_obs = len(group_sorted)
    y_series = group_sorted["y"].astype(float)
    price_series = group_sorted["price"].astype(float)

    # Lags relative to origin
    lag_1 = y_series.iloc[-1]
    lag_7 = y_series.iloc[-7] if n_obs >= 7 else y_series.iloc[0]
    lag_14 = y_series.iloc[-14] if n_obs >= 14 else y_series.iloc[0]
    lag_28 = y_series.iloc[-28] if n_obs >= 28 else y_series.iloc[0]

    # Rolling statistics
    roll_7 = y_series.iloc[-7:] if n_obs >= 7 else y_series
    roll_28 = y_series.iloc[-28:] if n_obs >= 28 else y_series

    roll_mean_7 = float(roll_7.mean())
    roll_std_7 = float(roll_7.std(ddof=0)) if len(roll_7) > 1 else 0.0
    roll_mean_28 = float(roll_28.mean())
    roll_std_28 = float(roll_28.std(ddof=0)) if len(roll_28) > 1 else 0.0

    # Days since last non-zero sale
    positive_sales = group_sorted[group_sorted["y"] > 0]
    if not positive_sales.empty:
        last_sale_date = positive_sales["date"].iloc[-1]
        days_since_sale = (origin - last_sale_date).days
    else:
        days_since_sale = 365

    # Price ratio relative to SKU median
    median_price = float(price_series.median()) if not price_series.empty else 0.0
    current_price = float(price_series.iloc[-1])
    price_ratio = (current_price / median_price) if median_price > 0 else 1.0

    return {
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
        "rolling_mean_7": roll_mean_7,
        "rolling_std_7": roll_std_7,
        "rolling_mean_28": roll_mean_28,
        "rolling_std_28": roll_std_28,
        "days_since_last_sale": days_since_sale,
        "price_ratio": price_ratio,
        "product_id": group_sorted["product_id"].iloc[-1],
        "abc_class": group_sorted["abc_class"].iloc[-1],
    }


def _build_target_day_row(
    sid: Any,
    origin: datetime.date,
    h: int,
    target_d: datetime.date,
    *,
    series_stats: dict[str, Any],
    cal_map: dict[datetime.date, Any],
    closure_windows: Sequence[tuple[datetime.date, datetime.date]],
    future_actuals: pd.DataFrame,
) -> dict[str, Any]:
    """Construct a direct-strategy row for a single horizon day."""
    cal_info = cal_map.get(target_d)
    if cal_info is not None:
        dow = int(cal_info.get("day_of_week", target_d.isocalendar()[2]))
        month = int(cal_info.get("month", target_d.month))
        iso_week = int(cal_info.get("iso_week", target_d.isocalendar()[1]))
        is_hol = bool(cal_info.get("is_holiday", False))
    else:
        dow = target_d.isocalendar()[2]
        month = target_d.month
        iso_week = target_d.isocalendar()[1]
        is_hol = False

    days_xmas = days_to_christmas(target_d)
    in_closure = is_in_closure_window(target_d, closure_windows)

    y_target: float | None = None
    if (sid, target_d) in future_actuals.index:
        actual_row = future_actuals.loc[(sid, target_d)]
        if isinstance(actual_row, pd.DataFrame):
            actual_row = actual_row.iloc[0]
        y_target = float(actual_row.get("y", 0.0))

    row = {
        "origin": origin,
        "series_id": sid,
        "horizon_day": h,
        "target_date": target_d,
        "target_day_of_week": dow,
        "target_month": month,
        "target_iso_week": iso_week,
        "target_is_holiday": is_hol,
        "target_days_to_christmas": days_xmas,
        "target_in_closure_window": in_closure,
        "y_target": y_target,
    }
    row.update(series_stats)
    return row


def build_direct_features_at_origin(
    history_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    origin: datetime.date,
    *,
    horizon: int = 28,
    closure_windows: Sequence[tuple[datetime.date, datetime.date]] | None = None,
) -> pd.DataFrame:
    """Extract direct-strategy feature rows at a specific origin date.

    Leakage-safe guarantee:
    - All features (lags, rolling stats, days since last sale, price ratio) are computed
      strictly from observations where date <= origin.
    - Future trading dates (origin < date <= origin + horizon) supply only target date
      calendar features and ground-truth targets (if available).
    """
    if closure_windows is None:
        closure_windows = default_christmas_closure_windows(origin.year - 1, origin.year + 1)

    hist = history_df[history_df["date"] <= origin].copy()
    if hist.empty:
        return pd.DataFrame()

    future_cal = (
        calendar_df[
            (calendar_df["date_key"] > origin)
            & (calendar_df["is_trading_day"].fillna(True).astype(bool))
        ]
        .sort_values("date_key")
        .head(horizon)
    )

    target_dates = list(future_cal["date_key"])
    if not target_dates:
        return pd.DataFrame()

    cal_map = {row["date_key"]: row for _, row in calendar_df.iterrows()}
    future_actuals = history_df[history_df["date"] > origin].set_index(["series_id", "date"])

    rows: list[dict[str, Any]] = []
    for sid, group in hist.groupby("series_id"):
        group_sorted = group.sort_values("date").reset_index(drop=True)
        if group_sorted.empty:
            continue

        series_stats = _compute_origin_series_stats(group_sorted, origin)
        for h, target_d in enumerate(target_dates, start=1):
            row = _build_target_day_row(
                sid=sid,
                origin=origin,
                h=h,
                target_d=target_d,
                series_stats=series_stats,
                cal_map=cal_map,
                closure_windows=closure_windows,
                future_actuals=future_actuals,
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_direct_training_dataset(
    series_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    *,
    horizon: int = 28,
    min_history_days: int = 35,
    sample_step_days: int = 7,
    closure_windows: Sequence[tuple[datetime.date, datetime.date]] | None = None,
) -> pd.DataFrame:
    """Build multi-origin training dataset for direct multi-step forecasting models.

    Per TRD §5.1:
    - Training rows are (origin, horizon_day h in 1..28) sampled weekly.
    - All features computed from data <= origin.
    - Target is y(origin + h).
    """
    trading_dates = sorted(series_df["date"].unique())
    if len(trading_dates) < min_history_days + horizon:
        return pd.DataFrame()

    first_date = trading_dates[0]
    min_origin = first_date + datetime.timedelta(days=min_history_days)
    max_target_date = trading_dates[-1]

    candidate_origins = [
        d for d in trading_dates if d >= min_origin and (max_target_date - d).days >= horizon
    ]
    if not candidate_origins:
        return pd.DataFrame()

    sampled_origins = candidate_origins[::sample_step_days]
    dataset_parts: list[pd.DataFrame] = []

    for origin in sampled_origins:
        part = build_direct_features_at_origin(
            history_df=series_df,
            calendar_df=calendar_df,
            origin=origin,
            horizon=horizon,
            closure_windows=closure_windows,
        )
        if not part.empty and part["y_target"].notna().all():
            dataset_parts.append(part)

    if not dataset_parts:
        return pd.DataFrame()

    return pd.concat(dataset_parts, ignore_index=True)
