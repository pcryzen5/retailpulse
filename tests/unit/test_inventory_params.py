"""Unit tests for synthetic inventory parameters and snapshot generation (P4-01)."""

from __future__ import annotations

import datetime
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.db.models.core import (
    DimDate,
    DimProduct,
    DimStore,
    FactDailySales,
    InventoryParam,
    InventorySnapshot,
)
from retailpulse.ml.inventory.params import (
    generate_synthetic_inventory_params,
    generate_synthetic_inventory_snapshots,
    seed_inventory_params,
)

pytestmark = pytest.mark.unit
runner = CliRunner()


@pytest.fixture
def in_memory_inventory_db() -> Session:
    """Create in-memory SQLite database seeded with products, store, date, and sales."""
    engine = sa.create_engine("sqlite:///:memory:")
    engine = engine.execution_options(
        schema_translate_map={"core": None, "ops": None, "ml": None, "app": None}
    )

    for tbl in [
        DimStore.__table__,
        DimProduct.__table__,
        DimDate.__table__,
        FactDailySales.__table__,
        InventoryParam.__table__,
        InventorySnapshot.__table__,
    ]:
        tbl.create(bind=engine, checkfirst=True)

    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    # Seed store
    store = DimStore(
        store_id=1, store_code="ONLINE", store_name="Online Store", country="United Kingdom"
    )
    session.add(store)

    # Seed products with different ABC classes
    products = [
        DimProduct(
            product_id=1,
            sku="SKU-A1",
            description="High value item A",
            is_product=True,
            abc_class="A",
            unit_cost_est=Decimal("50.00"),
        ),
        DimProduct(
            product_id=2,
            sku="SKU-A2",
            description="Fast mover item A",
            is_product=True,
            abc_class="A",
            unit_cost_est=Decimal("12.00"),
        ),
        DimProduct(
            product_id=3,
            sku="SKU-B1",
            description="Medium mover item B",
            is_product=True,
            abc_class="B",
            unit_cost_est=Decimal("20.00"),
        ),
        DimProduct(
            product_id=4,
            sku="SKU-C1",
            description="Slow mover item C",
            is_product=True,
            abc_class="C",
            unit_cost_est=Decimal("5.00"),
        ),
        DimProduct(
            product_id=5,
            sku="SKU-NONE",
            description="Unclassified product",
            is_product=True,
            abc_class=None,
            unit_cost_est=None,
        ),
    ]
    session.add_all(products)

    # Seed dates (last 30 days up to 2011-12-08)
    base_date = datetime.date(2011, 12, 8)
    dates = [
        DimDate(
            date_key=base_date - datetime.timedelta(days=i),
            year=2011,
            quarter=4,
            month=12,
            iso_week=49,
            day_of_week=1,
            day_of_month=8,
            is_weekend=False,
            is_holiday=False,
            is_trading_day=True,
            is_complete=True,
        )
        for i in range(35)
    ]
    session.add_all(dates)

    # Seed historical daily sales
    # Product 1: 14 units/day on 20 days with price 100.00 -> median price 100.00
    # Product 2: 7 units/day on 28 days with price 25.00 -> median price 25.00
    # Product 3: 1 unit/day on 14 days with price 35.00 -> median price 35.00
    # Product 4: 0 sales in window
    # Product 5: 0 sales in window
    sales: list[FactDailySales] = []
    for i in range(20):
        dt = base_date - datetime.timedelta(days=i)
        sales.append(
            FactDailySales(
                date_key=dt,
                store_id=1,
                product_id=1,
                units_sold=14,
                units_returned=0,
                revenue=Decimal("1400.00"),
                avg_unit_price=Decimal("100.00"),
                n_orders=5,
            )
        )
    for i in range(28):
        dt = base_date - datetime.timedelta(days=i)
        sales.append(
            FactDailySales(
                date_key=dt,
                store_id=1,
                product_id=2,
                units_sold=7,
                units_returned=0,
                revenue=Decimal("175.00"),
                avg_unit_price=Decimal("25.00"),
                n_orders=3,
            )
        )
    for i in range(14):
        dt = base_date - datetime.timedelta(days=i)
        sales.append(
            FactDailySales(
                date_key=dt,
                store_id=1,
                product_id=3,
                units_sold=1,
                units_returned=0,
                revenue=Decimal("35.00"),
                avg_unit_price=Decimal("35.00"),
                n_orders=1,
            )
        )
    session.add_all(sales)
    session.commit()
    return session


def test_synthetic_params_determinism(in_memory_inventory_db: Session) -> None:
    """Verify same seed produces identical parameters, and different seed changes outputs."""
    session = in_memory_inventory_db

    df1 = generate_synthetic_inventory_params(session=session, store_id=1, seed=42)
    df2 = generate_synthetic_inventory_params(session=session, store_id=1, seed=42)

    pd.testing.assert_frame_equal(df1, df2)

    # Different seed should generate some differing parameters
    df_diff = generate_synthetic_inventory_params(session=session, store_id=1, seed=999)
    assert not df1.equals(df_diff)


def test_synthetic_params_constraints_and_abc_rules(
    in_memory_inventory_db: Session,
) -> None:
    """Verify parameters obey ABC lead time rules, cost formulas, and DB constraints."""
    session = in_memory_inventory_db
    df = generate_synthetic_inventory_params(session=session, store_id=1, seed=42)

    assert len(df) == 5
    by_pid = {int(r["product_id"]): r for _, r in df.iterrows()}

    # Product 1 (Class A)
    p1 = by_pid[1]
    assert int(p1["lead_time_days"]) in {3, 7}
    assert p1["lead_time_std_days"] == Decimal(str(round(0.2 * float(p1["lead_time_days"]), 2)))
    assert p1["service_level"] == Decimal("0.980")
    assert p1["moq"] in {1, 6, 12}
    # Median price is 100.00 -> unit cost 0.6 * 100.0 = 60.00
    assert p1["unit_cost"] == Decimal("60.0000")
    assert p1["holding_cost_rate"] == Decimal("0.2000")
    assert p1["order_cost"] == Decimal("25.0000")
    assert p1["source"] == "synthetic"

    # Product 2 (Class A)
    p2 = by_pid[2]
    assert int(p2["lead_time_days"]) in {3, 7}
    assert p2["service_level"] == Decimal("0.980")
    # Median price is 25.00 -> unit cost 0.6 * 25.0 = 15.00
    assert p2["unit_cost"] == Decimal("15.0000")

    # Product 3 (Class B)
    p3 = by_pid[3]
    assert int(p3["lead_time_days"]) in {7, 14}
    assert p3["service_level"] == Decimal("0.950")
    # Median price is 35.00 -> unit cost 0.6 * 35.0 = 21.00
    assert p3["unit_cost"] == Decimal("21.0000")

    # Product 4 (Class C, no sales in window -> uses unit_cost_est = 5.00)
    p4 = by_pid[4]
    assert int(p4["lead_time_days"]) in {14, 21}
    assert p4["service_level"] == Decimal("0.900")
    assert p4["unit_cost"] == Decimal("5.0000")

    # Product 5 (Unclassified, no sales, no unit_cost_est -> fallback 6.00)
    p5 = by_pid[5]
    assert int(p5["lead_time_days"]) in {7, 14, 21}
    assert p5["service_level"] == Decimal("0.950")
    assert p5["unit_cost"] == Decimal("6.0000")


def test_synthetic_snapshots_baseline_rop(in_memory_inventory_db: Session) -> None:
    """Verify baseline ROP snapshot calculation: mu_d_28 * (L + 14)."""
    session = in_memory_inventory_db
    cutoff = datetime.date(2011, 12, 8)
    params_df = generate_synthetic_inventory_params(session=session, store_id=1, seed=42)

    snaps_df = generate_synthetic_inventory_snapshots(
        session=session,
        params_df=params_df,
        store_id=1,
        as_of=cutoff,
    )

    assert len(snaps_df) == 5
    by_pid = {int(r["product_id"]): r for _, r in snaps_df.iterrows()}

    # All snapshots must have source='synthetic', on_order=0, date=cutoff
    for pid in [1, 2, 3, 4, 5]:
        s = by_pid[pid]
        assert s["snapshot_date"] == cutoff
        assert s["store_id"] == 1
        assert s["on_order"] == 0
        assert s["source"] == "synthetic"
        assert s["on_hand"] >= 0

    # Product 2 had 7 units/day for all 28 days -> total 196 units -> daily avg = 7.0
    # Expected baseline reorder point calculation
    param_p2 = params_df[params_df["product_id"] == 2].iloc[0]
    expected_rop_p2 = round(7.0 * (float(param_p2["lead_time_days"]) + 14.0))
    assert by_pid[2]["on_hand"] == expected_rop_p2

    # Product 4 had 0 sales -> on_hand == 0
    assert by_pid[4]["on_hand"] == 0


def test_seed_inventory_params_idempotent_and_user_preservation(
    in_memory_inventory_db: Session,
) -> None:
    """Verify seeding persists rows, is idempotent, and preserves user-edited parameters."""
    session = in_memory_inventory_db
    cutoff = datetime.date(2011, 12, 8)

    # 1. First seeding run
    res1 = seed_inventory_params(session=session, store_id=1, as_of=cutoff, seed=42)
    assert res1.n_params_seeded == 5
    assert res1.n_snapshots_seeded == 5

    # Check rows in DB
    params_count = session.execute(sa.select(sa.func.count(InventoryParam.product_id))).scalar()
    snaps_count = session.execute(sa.select(sa.func.count(InventorySnapshot.product_id))).scalar()
    assert params_count == 5
    assert snaps_count == 5

    # 2. Modify one parameter to simulate user customization
    user_param = session.get(InventoryParam, (1, 1))
    assert user_param is not None
    user_param.lead_time_days = Decimal("5.0")
    user_param.source = "user"
    session.commit()

    # 3. Second seeding run with same seed
    res2 = seed_inventory_params(session=session, store_id=1, as_of=cutoff, seed=42)
    # Product 1 should NOT be overwritten because source='user'
    assert res2.n_params_seeded == 4
    assert res2.n_snapshots_seeded == 5

    # Verify total params count remains 5
    params_count_after = session.execute(
        sa.select(sa.func.count(InventoryParam.product_id))
    ).scalar()
    assert params_count_after == 5

    # Verify user param remained intact
    reloaded_user_param = session.get(InventoryParam, (1, 1))
    assert reloaded_user_param is not None
    assert reloaded_user_param.lead_time_days == Decimal("5.0")
    assert reloaded_user_param.source == "user"


def test_cli_seed_params_invocation(in_memory_inventory_db: Session, monkeypatch: Any) -> None:
    """Verify CLI retailpulse inventory seed-params command runs with JSON and table output."""
    session = in_memory_inventory_db

    # Monkeypatch session_scope in cli to yield our test in-memory session
    @contextmanager
    def mock_session_scope():  # type: ignore[no-untyped-def]
        yield session

    monkeypatch.setattr("retailpulse.cli.session_scope", mock_session_scope)

    # 1. Test JSON output
    result_json = runner.invoke(
        app,
        ["inventory", "seed-params", "--store-id", "1", "--as-of", "2011-12-08", "--json"],
    )
    assert result_json.exit_code == 0
    assert '"status": "succeeded"' in result_json.output
    assert '"n_params_seeded": 5' in result_json.output

    # 2. Test human-readable output
    result_text = runner.invoke(
        app,
        ["inventory", "seed-params", "--store-id", "1", "--as-of", "2011-12-08"],
    )
    assert result_text.exit_code == 0
    assert "Synthetic Inventory Parameters Seeded" in result_text.output
    assert "Notice: Parameters and snapshots are synthetic" in result_text.output

    # 3. Test invalid date format
    result_invalid = runner.invoke(
        app,
        ["inventory", "seed-params", "--as-of", "invalid-date"],
    )
    assert result_invalid.exit_code == 1
    assert "Invalid date format" in result_invalid.output
