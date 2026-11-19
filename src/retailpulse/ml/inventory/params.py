"""Synthetic inventory parameter and initial snapshot generation.

Implements deterministic parameter generation for inventory optimization
(TRD §5.4, PRD §7.5 FR-INV-01):
  - Lead time L in {3, 7, 14, 21} determined by product ABC class.
  - Lead time standard deviation sigma_LT = 0.2 * L.
  - Minimum Order Quantity (MOQ) deterministically drawn from {1, 6, 12}.
  - Unit cost C = 0.6 * median selling price (floored at 0.01).
  - Holding cost rate h = 0.20 (20% annual).
  - Order cost K = 25.00 ($25 per order).
  - Service level z: class A = 0.980, class B = 0.950, class C = 0.900 (default 0.950).
  - Tagged source = 'synthetic'.

Initial inventory snapshot:
  - Baseline ROP = 28-day moving average daily sales * (L + 14).
  - Initial on_hand = max(0, round(ROP_baseline)).
  - on_order = 0.
  - Tagged source = 'synthetic'.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.core import (
    DimDate,
    DimProduct,
    FactDailySales,
    InventoryParam,
    InventorySnapshot,
)


@dataclass(frozen=True)
class InventorySeedResult:
    """Result summary of seeding synthetic inventory parameters and snapshots."""

    store_id: int
    data_as_of: datetime.date
    n_params_seeded: int
    n_snapshots_seeded: int
    seed: int
    source: str = "synthetic"


def _resolve_as_of_date(session: Session, as_of: datetime.date | None) -> datetime.date:
    """Resolve the reference as-of date from DimDate or FactDailySales."""
    if as_of is not None:
        return as_of

    latest_date = session.execute(
        sa.select(sa.func.max(DimDate.date_key)).where(DimDate.is_complete.is_(True))
    ).scalar_one_or_none()
    if latest_date is not None:
        return latest_date

    max_sales_date = session.execute(
        sa.select(sa.func.max(FactDailySales.date_key))
    ).scalar_one_or_none()
    if max_sales_date is not None:
        return max_sales_date

    return datetime.date(2011, 12, 8)


def _get_product_median_prices(
    session: Session,
    store_id: int,
    product_ids: Sequence[int],
) -> dict[int, float]:
    """Calculate median unit selling price per product from historical daily sales."""
    if not product_ids:
        return {}

    stmt = sa.select(
        FactDailySales.product_id,
        FactDailySales.avg_unit_price,
    ).where(
        FactDailySales.store_id == store_id,
        FactDailySales.product_id.in_(product_ids),
        FactDailySales.avg_unit_price.is_not(None),
        FactDailySales.avg_unit_price > 0,
    )
    rows = session.execute(stmt).all()

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=["product_id", "avg_unit_price"])
    df["avg_unit_price"] = df["avg_unit_price"].astype(float)
    medians = df.groupby("product_id")["avg_unit_price"].median().to_dict()

    return {int(pid): float(val) for pid, val in medians.items()}


def generate_synthetic_inventory_params(
    session: Session,
    store_id: int = 1,
    seed: int = 42,
    product_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Generate deterministic synthetic inventory policy parameters for products.

    Args:
        session: Active database session.
        store_id: Target store identifier.
        seed: Random seed for deterministic generation.
        product_ids: Optional list of product IDs to restrict to. If None, uses all active products.

    Returns:
        DataFrame with columns corresponding to core.inventory_param attributes.
    """
    stmt = (
        sa.select(
            DimProduct.product_id,
            DimProduct.sku,
            DimProduct.abc_class,
            DimProduct.unit_cost_est,
        )
        .where(DimProduct.is_product.is_(True))
        .order_by(DimProduct.product_id.asc())
    )
    if product_ids is not None:
        stmt = stmt.where(DimProduct.product_id.in_(product_ids))

    products = session.execute(stmt).all()
    if not products:
        return pd.DataFrame(
            columns=[
                "product_id",
                "store_id",
                "lead_time_days",
                "lead_time_std_days",
                "service_level",
                "unit_cost",
                "holding_cost_rate",
                "order_cost",
                "moq",
                "source",
            ]
        )

    all_pids = [int(p.product_id) for p in products]
    median_prices = _get_product_median_prices(session, store_id, all_pids)

    rows: list[dict[str, Any]] = []
    for prod in products:
        pid = int(prod.product_id)
        abc = str(prod.abc_class).strip().upper() if prod.abc_class else None

        # Deterministic per-product pseudo-random number generator
        p_seed = int((seed * 1_000_003 + pid * 10_007) % (2**31 - 1))
        rng = np.random.default_rng(p_seed)

        # 1. Lead time L in {3, 7, 14, 21} by ABC class
        if abc == "A":
            lt_choices = [3, 7]
            service_level = Decimal("0.980")
        elif abc == "B":
            lt_choices = [7, 14]
            service_level = Decimal("0.950")
        elif abc == "C":
            lt_choices = [14, 21]
            service_level = Decimal("0.900")
        else:
            lt_choices = [7, 14, 21]
            service_level = Decimal("0.950")

        lead_time = int(rng.choice(lt_choices))
        lead_time_days = Decimal(str(lead_time))

        # 2. Lead time std: sigma_LT = 0.2 * L
        lead_time_std_days = Decimal(str(round(0.2 * lead_time, 2)))

        # 3. Minimum Order Quantity (MOQ) from {1, 6, 12}
        moq_choices = [1, 6, 12]
        moq = int(rng.choice(moq_choices))

        # 4. Unit cost C = 0.6 * median selling price (floored at 0.01)
        med_price = median_prices.get(pid)
        if med_price is None or med_price <= 0:
            if prod.unit_cost_est is not None and prod.unit_cost_est > 0:
                cost_val = float(prod.unit_cost_est)
            else:
                cost_val = 6.00
        else:
            cost_val = 0.6 * med_price
        unit_cost = max(Decimal("0.0100"), Decimal(str(round(cost_val, 4))))

        # 5. Holding cost rate & order cost
        holding_cost_rate = Decimal("0.2000")
        order_cost = Decimal("25.0000")

        rows.append(
            {
                "product_id": pid,
                "store_id": store_id,
                "lead_time_days": lead_time_days,
                "lead_time_std_days": lead_time_std_days,
                "service_level": service_level,
                "unit_cost": unit_cost,
                "holding_cost_rate": holding_cost_rate,
                "order_cost": order_cost,
                "moq": moq,
                "source": "synthetic",
            }
        )

    return pd.DataFrame(rows)


def generate_synthetic_inventory_snapshots(
    session: Session,
    params_df: pd.DataFrame,
    store_id: int = 1,
    as_of: datetime.date | None = None,
) -> pd.DataFrame:
    """Generate initial inventory snapshots based on baseline ROP policy.

    Baseline policy formula (TRD §5.4):
      ROP_baseline = (28-day moving average daily sales) * (lead_time_days + 14)
      initial on_hand = max(0, round(ROP_baseline))
      on_order = 0

    Args:
        session: Active database session.
        params_df: Generated or existing inventory parameters DataFrame.
        store_id: Target store identifier.
        as_of: Snapshot date (defaults to latest complete date).

    Returns:
        DataFrame with columns corresponding to core.inventory_snapshot attributes.
    """
    if params_df.empty:
        return pd.DataFrame(
            columns=["snapshot_date", "store_id", "product_id", "on_hand", "on_order", "source"]
        )

    effective_as_of = _resolve_as_of_date(session, as_of)
    start_date = effective_as_of - datetime.timedelta(days=27)  # 28-day window inclusive

    product_ids = [int(p) for p in params_df["product_id"].unique()]

    # Query 28-day total sales per product
    stmt = (
        sa.select(
            FactDailySales.product_id,
            sa.func.coalesce(sa.func.sum(FactDailySales.units_sold), 0).label("units_28d"),
        )
        .where(
            FactDailySales.store_id == store_id,
            FactDailySales.product_id.in_(product_ids),
            FactDailySales.date_key >= start_date,
            FactDailySales.date_key <= effective_as_of,
        )
        .group_by(FactDailySales.product_id)
    )
    sales_rows = session.execute(stmt).all()
    sales_map = {int(r.product_id): int(r.units_28d) for r in sales_rows}

    # Map parameters to calculate baseline ROP
    rows: list[dict[str, Any]] = []
    for _, param in params_df.iterrows():
        pid = int(param["product_id"])
        lt = float(param["lead_time_days"])

        total_units = sales_map.get(pid, 0)
        daily_avg = max(0.0, total_units / 28.0)
        rop_baseline = daily_avg * (lt + 14.0)
        on_hand = max(0, round(rop_baseline))

        rows.append(
            {
                "snapshot_date": effective_as_of,
                "store_id": store_id,
                "product_id": pid,
                "on_hand": on_hand,
                "on_order": 0,
                "source": "synthetic",
            }
        )

    return pd.DataFrame(rows)


def seed_inventory_params(
    session: Session,
    store_id: int = 1,
    as_of: datetime.date | None = None,
    seed: int | None = None,
    product_ids: Sequence[int] | None = None,
) -> InventorySeedResult:
    """Generate and persist synthetic inventory parameters and snapshots idempotently.

    Args:
        session: Active SQLAlchemy database session.
        store_id: Target store identifier.
        as_of: Effective cutoff date for initial snapshot.
        seed: Random seed (defaults to config RP_SEED).
        product_ids: Optional list of product IDs to restrict to.

    Returns:
        InventorySeedResult with counts of seeded records.
    """
    settings = get_settings()
    eff_seed = seed if seed is not None else settings.seed
    effective_as_of = _resolve_as_of_date(session, as_of)

    # 1. Generate parameters
    params_df = generate_synthetic_inventory_params(
        session=session,
        store_id=store_id,
        seed=eff_seed,
        product_ids=product_ids,
    )

    if params_df.empty:
        return InventorySeedResult(
            store_id=store_id,
            data_as_of=effective_as_of,
            n_params_seeded=0,
            n_snapshots_seeded=0,
            seed=eff_seed,
        )

    # 2. Generate initial snapshots
    snapshots_df = generate_synthetic_inventory_snapshots(
        session=session,
        params_df=params_df,
        store_id=store_id,
        as_of=effective_as_of,
    )

    pids = [int(p) for p in params_df["product_id"].unique()]

    # 3. Persist atomically
    # Delete existing synthetic parameters for targeted products and store
    session.execute(
        sa.delete(InventoryParam).where(
            InventoryParam.store_id == store_id,
            InventoryParam.product_id.in_(pids),
            InventoryParam.source == "synthetic",
        )
    )

    # Check which products already have non-synthetic (e.g. 'user') parameters
    existing_user_pids = set(
        session.execute(
            sa.select(InventoryParam.product_id).where(
                InventoryParam.store_id == store_id,
                InventoryParam.product_id.in_(pids),
                InventoryParam.source != "synthetic",
            )
        )
        .scalars()
        .all()
    )

    # Filter out parameters for products that have user overrides
    params_to_insert = [
        InventoryParam(
            product_id=int(row["product_id"]),
            store_id=int(row["store_id"]),
            lead_time_days=row["lead_time_days"],
            lead_time_std_days=row["lead_time_std_days"],
            service_level=row["service_level"],
            unit_cost=row["unit_cost"],
            holding_cost_rate=row["holding_cost_rate"],
            order_cost=row["order_cost"],
            moq=int(row["moq"]),
            source=str(row["source"]),
        )
        for _, row in params_df.iterrows()
        if int(row["product_id"]) not in existing_user_pids
    ]
    session.add_all(params_to_insert)

    # Delete existing snapshots for targeted date, store, and products
    session.execute(
        sa.delete(InventorySnapshot).where(
            InventorySnapshot.snapshot_date == effective_as_of,
            InventorySnapshot.store_id == store_id,
            InventorySnapshot.product_id.in_(pids),
        )
    )

    snapshots_to_insert = [
        InventorySnapshot(
            snapshot_date=row["snapshot_date"],
            store_id=int(row["store_id"]),
            product_id=int(row["product_id"]),
            on_hand=int(row["on_hand"]),
            on_order=int(row["on_order"]),
            source=str(row["source"]),
        )
        for _, row in snapshots_df.iterrows()
    ]
    session.add_all(snapshots_to_insert)

    session.commit()

    return InventorySeedResult(
        store_id=store_id,
        data_as_of=effective_as_of,
        n_params_seeded=len(params_to_insert),
        n_snapshots_seeded=len(snapshots_to_insert),
        seed=eff_seed,
        source="synthetic",
    )
