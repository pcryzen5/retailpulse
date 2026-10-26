"""Aggregation of sales lines into daily product/store summaries and product classification.

Implements P1-09, DATABASE.md §11, and TRD §4.3 & §4.4:
- fact_daily_sales for touched dates (excluding outlier lines and incomplete days)
- Product first_sale_date and last_sale_date derivation
- Customer first_order_date and last_order_date derivation
- ABC product classification (trailing-365-day revenue: A <= 80% cumulative, B <= 95%, C rest)
- XYZ product classification (coefficient of variation of weekly demand: X < 0.5, Y < 1.0, Z rest)
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.core import (
    DimCustomer,
    DimDate,
    DimProduct,
    FactDailySales,
    FactSalesLine,
)
from retailpulse.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AggregationResult:
    """Summary metrics from executing the daily sales aggregation and classification."""

    dates_processed: int
    daily_records_written: int
    products_updated: int
    abc_counts: dict[str, int]
    xyz_counts: dict[str, int]
    data_as_of: datetime.date | None


def aggregate_daily_sales(
    session: Session,
    dates: Iterable[datetime.date] | None = None,
    *,
    batch_id: uuid.UUID | None = None,
) -> int:
    """Recompute fact_daily_sales for specified or touched dates.

    Per DATABASE.md §11 & TRD §4.4:
    - Only recompute dates touched by the batch/request.
    - Exclude outlier lines (is_outlier = true).
    - Exclude incomplete days (dim_date.is_complete = false).
    - Exclude non-product SKUs (dim_product.is_product = false).
    - Sparse: only (date, store, product) tuples with activity (sales or returns) are stored.
    - Idempotent: re-aggregating the same dates replaces existing fact_daily_sales rows.

    Args:
        session: Active SQLAlchemy database session.
        dates: Explicit collection of calendar dates to recompute. If None, derives
            dates from batch_id or all distinct dates in fact_sales_line.
        batch_id: Optional ingestion batch UUID used to identify touched dates if
            dates is None.

    Returns:
        Number of FactDailySales records written/upserted.
    """
    touched_dates = _resolve_touched_dates(session, dates=dates, batch_id=batch_id)
    if not touched_dates:
        logger.info("no touched dates found for aggregation; skipping")
        return 0

    # 1. Determine which touched dates are complete
    complete_stmt = sa.select(DimDate.date_key).where(
        DimDate.date_key.in_(touched_dates),
        DimDate.is_complete.is_(True),
    )
    complete_dates = set(session.execute(complete_stmt).scalars().all())

    # 2. Clear existing daily sales rows for ALL touched dates to ensure idempotency
    # and guarantee that any newly incomplete dates have 0 daily sales records.
    del_stmt = sa.delete(FactDailySales).where(FactDailySales.date_key.in_(touched_dates))
    session.execute(del_stmt)

    if not complete_dates:
        session.flush()
        logger.info(
            "no complete dates among touched dates for daily aggregation",
            total_touched=len(touched_dates),
        )
        return 0

    # 3. Aggregate non-outlier product sales lines on complete dates
    stmt = (
        sa.select(
            FactSalesLine.date_key,
            FactSalesLine.store_id,
            FactSalesLine.product_id,
            sa.func.coalesce(
                sa.func.sum(
                    sa.case(
                        (
                            sa.and_(
                                FactSalesLine.is_return.is_(False),
                                FactSalesLine.quantity > 0,
                            ),
                            FactSalesLine.quantity,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("units_sold"),
            sa.func.coalesce(
                sa.func.sum(
                    sa.case(
                        (
                            sa.and_(
                                FactSalesLine.is_return.is_(True),
                                FactSalesLine.quantity < 0,
                            ),
                            sa.func.abs(FactSalesLine.quantity),
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("units_returned"),
            sa.func.coalesce(
                sa.func.sum(
                    sa.case(
                        (
                            FactSalesLine.is_return.is_(False),
                            FactSalesLine.quantity * FactSalesLine.unit_price,
                        ),
                        else_=0,
                    )
                ),
                0,
            ).label("revenue"),
            sa.func.count(
                sa.func.distinct(
                    sa.case(
                        (
                            FactSalesLine.is_return.is_(False),
                            FactSalesLine.invoice_no,
                        ),
                        else_=None,
                    )
                )
            ).label("n_orders"),
        )
        .join(DimProduct, FactSalesLine.product_id == DimProduct.product_id)
        .where(
            FactSalesLine.date_key.in_(complete_dates),
            FactSalesLine.is_outlier.is_(False),
            DimProduct.is_product.is_(True),
        )
        .group_by(FactSalesLine.date_key, FactSalesLine.store_id, FactSalesLine.product_id)
    )

    rows = session.execute(stmt).all()
    if not rows:
        session.flush()
        logger.info("no sales activity to aggregate on complete touched dates")
        return 0

    records: list[FactDailySales] = []
    four_places = Decimal("0.0001")

    for r in rows:
        u_sold = int(r.units_sold)
        u_returned = int(r.units_returned)

        # Sparse storage: skip rows that have zero demand and zero returns
        if u_sold == 0 and u_returned == 0:
            continue

        raw_rev = Decimal(str(r.revenue)) if r.revenue is not None else Decimal(0)
        revenue = raw_rev.quantize(four_places)

        avg_unit_price: Decimal | None = None
        if u_sold > 0:
            avg_unit_price = (revenue / Decimal(u_sold)).quantize(four_places)

        records.append(
            FactDailySales(
                date_key=r.date_key,
                store_id=r.store_id,
                product_id=r.product_id,
                units_sold=u_sold,
                units_returned=u_returned,
                revenue=revenue,
                avg_unit_price=avg_unit_price,
                n_orders=int(r.n_orders),
            )
        )

    session.add_all(records)
    session.flush()

    logger.info(
        "aggregated daily sales complete",
        touched_dates=len(touched_dates),
        complete_dates=len(complete_dates),
        daily_records=len(records),
    )
    return len(records)


def compute_product_first_last_dates(
    session: Session,
    product_ids: Iterable[int] | None = None,
) -> int:
    """Update first_sale_date and last_sale_date on dim_product from sales lines.

    Calculates the earliest and latest sales transaction date per product from
    non-outlier, non-return sales lines.

    Args:
        session: Active SQLAlchemy database session.
        product_ids: Optional collection of product_ids to restrict update to.
            If None, all active products are re-evaluated.

    Returns:
        Number of product dimension records updated.
    """
    stmt = (
        sa.select(
            FactSalesLine.product_id,
            sa.func.min(FactSalesLine.date_key).label("first_date"),
            sa.func.max(FactSalesLine.date_key).label("last_date"),
        )
        .where(
            FactSalesLine.is_outlier.is_(False),
            FactSalesLine.is_return.is_(False),
        )
        .group_by(FactSalesLine.product_id)
    )

    if product_ids is not None:
        stmt = stmt.where(FactSalesLine.product_id.in_(product_ids))

    results = session.execute(stmt).all()
    if not results:
        return 0

    date_map = {r.product_id: (r.first_date, r.last_date) for r in results}

    prod_stmt = sa.select(DimProduct).where(DimProduct.product_id.in_(date_map.keys()))
    products = session.execute(prod_stmt).scalars().all()

    updated = 0
    for p in products:
        f_date, l_date = date_map[p.product_id]
        if p.first_sale_date != f_date or p.last_sale_date != l_date:
            p.first_sale_date = f_date
            p.last_sale_date = l_date
            updated += 1

    session.flush()
    logger.info("updated product sale date boundaries", updated=updated)
    return updated


def compute_customer_order_dates(
    session: Session,
    customer_ids: Iterable[int] | None = None,
) -> int:
    """Update first_order_date and last_order_date on dim_customer from sales lines.

    Args:
        session: Active SQLAlchemy database session.
        customer_ids: Optional collection of customer_ids to restrict update to.
            If None, all customers with orders are re-evaluated.

    Returns:
        Number of customer dimension records updated.
    """
    stmt = (
        sa.select(
            FactSalesLine.customer_id,
            sa.func.min(FactSalesLine.date_key).label("first_date"),
            sa.func.max(FactSalesLine.date_key).label("last_date"),
        )
        .where(
            FactSalesLine.customer_id.is_not(None),
            FactSalesLine.is_outlier.is_(False),
        )
        .group_by(FactSalesLine.customer_id)
    )

    if customer_ids is not None:
        stmt = stmt.where(FactSalesLine.customer_id.in_(customer_ids))

    results = session.execute(stmt).all()
    if not results:
        return 0

    date_map = {r.customer_id: (r.first_date, r.last_date) for r in results}

    cust_stmt = sa.select(DimCustomer).where(DimCustomer.customer_id.in_(date_map.keys()))
    customers = session.execute(cust_stmt).scalars().all()

    updated = 0
    for c in customers:
        f_date, l_date = date_map[c.customer_id]
        if c.first_order_date != f_date or c.last_order_date != l_date:
            c.first_order_date = f_date
            c.last_order_date = l_date
            updated += 1

    session.flush()
    logger.info("updated customer order date boundaries", updated=updated)
    return updated


def classify_abc_xyz(
    session: Session,
    as_of: datetime.date | None = None,
) -> dict[str, int]:
    """Recompute ABC and XYZ classifications for products in dim_product.

    Per DATABASE.md §11:
    - ABC: Trailing 365 days revenue (A <= 80% cumulative, B <= 95%, C rest).
    - XYZ: Coefficient of variation (std / mean) of weekly demand:
      X < 0.5, Y < 1.0, Z otherwise.
    - Non-product SKUs (is_product = false) have abc_class and xyz_class set to None.

    Args:
        session: Active SQLAlchemy database session.
        as_of: Reference date representing data_as_of. If None, uses the latest
            complete date in fact_daily_sales.

    Returns:
        Dictionary of category counts, e.g. {"A": 10, "B": 25, "C": 100, "X": 15, "Y": 30, "Z": 90}.
    """
    if as_of is None:
        max_date_stmt = sa.select(sa.func.max(FactDailySales.date_key))
        as_of = session.execute(max_date_stmt).scalar()

    counts = {"A": 0, "B": 0, "C": 0, "X": 0, "Y": 0, "Z": 0}
    if as_of is None:
        logger.info("no sales data found to compute ABC/XYZ classes")
        return counts

    # Trailing 365 calendar days [start_date, as_of]
    start_date = as_of - datetime.timedelta(days=364)

    # 1. Fetch all product records
    prod_stmt = sa.select(DimProduct)
    all_products = session.execute(prod_stmt).scalars().all()
    # Reset non-products to None
    for p in all_products:
        if not p.is_product:
            p.abc_class = None
            p.xyz_class = None

    eligible_products = [p for p in all_products if p.is_product]
    if not eligible_products:
        session.flush()
        return counts

    # 2. ABC Classification based on trailing-365-day revenue
    abc_classes = _compute_abc_classes(
        session,
        eligible_products=eligible_products,
        start_date=start_date,
        as_of=as_of,
    )

    # 3. XYZ Classification based on coefficient of variation of weekly demand
    xyz_classes = _compute_xyz_classes(
        session,
        eligible_products=eligible_products,
        start_date=start_date,
        as_of=as_of,
    )

    # 4. Assign classes back to DimProduct
    for p in eligible_products:
        abc = abc_classes.get(p.product_id, "C")
        xyz = xyz_classes.get(p.product_id, "Z")
        p.abc_class = abc
        p.xyz_class = xyz
        counts[abc] += 1
        counts[xyz] += 1

    session.flush()
    logger.info("ABC/XYZ product classification completed", as_of=str(as_of), counts=counts)
    return counts


def _compute_abc_classes(
    session: Session,
    *,
    eligible_products: list[DimProduct],
    start_date: datetime.date,
    as_of: datetime.date,
) -> dict[int, str]:
    """Compute ABC classification mapping from trailing daily sales revenue."""
    rev_stmt = (
        sa.select(
            FactDailySales.product_id,
            sa.func.coalesce(sa.func.sum(FactDailySales.revenue), 0).label("tot_rev"),
        )
        .where(
            FactDailySales.date_key >= start_date,
            FactDailySales.date_key <= as_of,
        )
        .group_by(FactDailySales.product_id)
    )

    rev_rows = session.execute(rev_stmt).all()
    rev_map = {r.product_id: float(r.tot_rev) for r in rev_rows}

    # Sort eligible products descending by revenue
    sorted_prods = sorted(
        eligible_products,
        key=lambda p: rev_map.get(p.product_id, 0.0),
        reverse=True,
    )

    total_revenue = sum(rev_map.get(p.product_id, 0.0) for p in sorted_prods)

    abc_classes: dict[int, str] = {}
    if total_revenue <= 0.0:
        # All products default to C if no positive revenue
        for p in eligible_products:
            abc_classes[p.product_id] = "C"
        return abc_classes

    cum_rev = 0.0
    for p in sorted_prods:
        rev = rev_map.get(p.product_id, 0.0)
        if rev <= 0.0:
            abc_classes[p.product_id] = "C"
            continue

        prior_cum_share = cum_rev / total_revenue
        if prior_cum_share < 0.80:
            abc_classes[p.product_id] = "A"
        elif prior_cum_share < 0.95:
            abc_classes[p.product_id] = "B"
        else:
            abc_classes[p.product_id] = "C"

        cum_rev += rev

    return abc_classes


def _compute_xyz_classes(
    session: Session,
    *,
    eligible_products: list[DimProduct],
    start_date: datetime.date,
    as_of: datetime.date,
) -> dict[int, str]:
    """Compute XYZ classification mapping from weekly demand variation."""
    # Retrieve all distinct calendar weeks in the trailing window
    week_stmt = (
        sa.select(DimDate.year, DimDate.iso_week)
        .where(
            DimDate.date_key >= start_date,
            DimDate.date_key <= as_of,
        )
        .distinct()
        .order_by(DimDate.year, DimDate.iso_week)
    )
    all_weeks = session.execute(week_stmt).all()
    num_weeks = len(all_weeks)

    xyz_classes: dict[int, str] = {}
    if num_weeks == 0:
        for p in eligible_products:
            xyz_classes[p.product_id] = "Z"
        return xyz_classes

    # Sum units_sold per (product, year, iso_week)
    weekly_stmt = (
        sa.select(
            FactDailySales.product_id,
            DimDate.year,
            DimDate.iso_week,
            sa.func.sum(FactDailySales.units_sold).label("weekly_units"),
        )
        .join(DimDate, FactDailySales.date_key == DimDate.date_key)
        .where(
            FactDailySales.date_key >= start_date,
            FactDailySales.date_key <= as_of,
        )
        .group_by(FactDailySales.product_id, DimDate.year, DimDate.iso_week)
    )

    weekly_rows = session.execute(weekly_stmt).all()

    # Build per-product week demands: {product_id: {(year, week): units}}
    prod_weekly: dict[int, dict[tuple[int, int], int]] = {
        p.product_id: {} for p in eligible_products
    }
    for r in weekly_rows:
        pid = r.product_id
        if pid in prod_weekly:
            prod_weekly[pid][(r.year, r.iso_week)] = int(r.weekly_units or 0)

    for p in eligible_products:
        week_dict = prod_weekly.get(p.product_id, {})
        # Zero-fill weeks where the product had no recorded sales
        demand_series = np.array([float(week_dict.get(w, 0)) for w in all_weeks], dtype=float)

        mean_demand = float(np.mean(demand_series))
        if mean_demand <= 0.0:
            xyz_classes[p.product_id] = "Z"
            continue

        # Sample standard deviation (ddof=1 if N > 1, else 0)
        std_demand = float(np.std(demand_series, ddof=1 if num_weeks > 1 else 0))
        cv = std_demand / mean_demand

        if cv < 0.5:
            xyz_classes[p.product_id] = "X"
        elif cv < 1.0:
            xyz_classes[p.product_id] = "Y"
        else:
            xyz_classes[p.product_id] = "Z"

    return xyz_classes


def run_aggregation_pipeline(
    session: Session,
    dates: Iterable[datetime.date] | None = None,
    *,
    batch_id: uuid.UUID | None = None,
    as_of: datetime.date | None = None,
) -> AggregationResult:
    """Execute complete post-ingestion aggregation pipeline.

    1. Aggregates FactDailySales for touched complete dates.
    2. Updates DimProduct first/last sales dates.
    3. Updates DimCustomer first/last order dates.
    4. Computes ABC and XYZ classifications for all products.

    Args:
        session: Active database session.
        dates: Explicit dates touched by the load.
        batch_id: Batch UUID to identify touched dates.
        as_of: Optional reference date for classification window.

    Returns:
        AggregationResult summary.
    """
    touched = _resolve_touched_dates(session, dates=dates, batch_id=batch_id)
    daily_count = aggregate_daily_sales(session, dates=touched)
    prods_updated = compute_product_first_last_dates(session)
    compute_customer_order_dates(session)

    classes = classify_abc_xyz(session, as_of=as_of)

    if as_of is None:
        max_d_stmt = sa.select(sa.func.max(FactDailySales.date_key))
        as_of = session.execute(max_d_stmt).scalar()

    abc_counts = {k: classes.get(k, 0) for k in ("A", "B", "C")}
    xyz_counts = {k: classes.get(k, 0) for k in ("X", "Y", "Z")}

    return AggregationResult(
        dates_processed=len(touched),
        daily_records_written=daily_count,
        products_updated=prods_updated,
        abc_counts=abc_counts,
        xyz_counts=xyz_counts,
        data_as_of=as_of,
    )


def _resolve_touched_dates(
    session: Session,
    dates: Iterable[datetime.date] | None = None,
    batch_id: uuid.UUID | None = None,
) -> set[datetime.date]:
    """Resolve distinct dates touched by dates argument, batch_id, or whole table."""
    if dates is not None:
        return set(dates)

    if batch_id is not None:
        stmt = (
            sa.select(FactSalesLine.date_key).where(FactSalesLine.batch_id == batch_id).distinct()
        )
        return set(session.execute(stmt).scalars().all())

    # Fallback: all distinct dates currently present in fact_sales_line
    stmt = sa.select(FactSalesLine.date_key).distinct()
    return set(session.execute(stmt).scalars().all())
