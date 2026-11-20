"""Unit and integration tests for inventory replenishment recommendations (P4-03).

Verifies:
  - Recommendation generation using point forecasts, backtest RMSE (sigma_e), params, and snapshots.
  - Full coverage of all four inventory statuses: ('reorder_now', 'overstock', 'ok', 'no_data').
  - Atomic pointer update in ml.active_output for task='inventory'.
  - Idempotent re-runs.
  - CLI retailpulse score inventory command execution (JSON and text modes).
"""

from __future__ import annotations

import datetime
import uuid
from contextlib import contextmanager
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.db.models.core import (
    DimDate,
    DimProduct,
    DimStore,
    InventoryParam,
    InventorySnapshot,
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    Forecast,
    ForecastMetric,
    InventoryRecommendation,
    ModelRun,
)
from retailpulse.ml.inventory.recommend import (
    TASK_FORECAST,
    TASK_INVENTORY,
    generate_inventory_recommendations,
)

pytestmark = pytest.mark.unit
runner = CliRunner()


@pytest.fixture
def in_memory_recommend_db() -> Session:
    """Create in-memory SQLite database seeded with 4 products covering all four statuses."""
    engine = sa.create_engine("sqlite:///:memory:")
    engine = engine.execution_options(
        schema_translate_map={"core": None, "ops": None, "ml": None, "app": None}
    )

    for tbl in [
        DimStore.__table__,
        DimProduct.__table__,
        DimDate.__table__,
        InventoryParam.__table__,
        InventorySnapshot.__table__,
        ModelRun.__table__,
        ForecastMetric.__table__,
        Forecast.__table__,
        InventoryRecommendation.__table__,
        ActiveOutput.__table__,
    ]:
        tbl.create(bind=engine, checkfirst=True)

    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    # 1. Store
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store", country="UK")
    session.add(store)

    # 2. Date
    cutoff = datetime.date(2011, 12, 8)
    dates = [
        DimDate(
            date_key=cutoff - datetime.timedelta(days=i),
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
        for i in range(10)
    ]
    session.add_all(dates)

    # 3. Model run for forecast (champion)
    forecast_run_id = uuid.uuid4()
    run = ModelRun(
        run_id=forecast_run_id,
        task=TASK_FORECAST,
        algorithm="lightgbm",
        mlflow_run_id="mlflow-forecast-123",
        status="succeeded",
        is_champion=True,
        params={},
        metrics={},
    )
    session.add(run)

    # Active output for forecast
    active_fc = ActiveOutput(
        task=TASK_FORECAST,
        model_run_id=forecast_run_id,
        data_as_of=cutoff,
    )
    session.add(active_fc)

    # 4. Forecast error metrics (RMSE per ABC and global)
    metrics = [
        ForecastMetric(
            model_run_id=forecast_run_id,
            scope="abc",
            scope_key="A",
            metric="rmse",
            value=2.0,
        ),
        ForecastMetric(
            model_run_id=forecast_run_id,
            scope="abc",
            scope_key="B",
            metric="rmse",
            value=1.5,
        ),
        ForecastMetric(
            model_run_id=forecast_run_id,
            scope="global",
            scope_key="all",
            metric="rmse",
            value=1.8,
        ),
    ]
    session.add_all(metrics)

    # 5. Products designed for each status:
    # Product 1: reorder_now (IP = 10, ROP ~ 94, IP <= ROP)
    # Product 2: overstock (IP = 600, daily sales = 5 -> days of cover = 120 > 90)
    # Product 3: ok (IP = 150, daily sales = 5 -> days of cover = 30 <= 90)
    # Product 4: no_data (no forecast rows)
    products = [
        DimProduct(
            product_id=1, sku="SKU-REORDER", description="Item 1", abc_class="A", is_product=True
        ),
        DimProduct(
            product_id=2, sku="SKU-OVERSTOCK", description="Item 2", abc_class="B", is_product=True
        ),
        DimProduct(
            product_id=3, sku="SKU-OK", description="Item 3", abc_class="B", is_product=True
        ),
        DimProduct(
            product_id=4, sku="SKU-NODATA", description="Item 4", abc_class="C", is_product=True
        ),
    ]
    session.add_all(products)

    # 6. Inventory params
    params = [
        InventoryParam(
            product_id=1,
            store_id=1,
            lead_time_days=Decimal("7.0"),
            lead_time_std_days=Decimal("1.40"),
            service_level=Decimal("0.950"),
            unit_cost=Decimal("10.0000"),
            holding_cost_rate=Decimal("0.2000"),
            order_cost=Decimal("25.00"),
            moq=6,
            source="synthetic",
        ),
        InventoryParam(
            product_id=2,
            store_id=1,
            lead_time_days=Decimal("7.0"),
            lead_time_std_days=Decimal("1.40"),
            service_level=Decimal("0.950"),
            unit_cost=Decimal("10.0000"),
            holding_cost_rate=Decimal("0.2000"),
            order_cost=Decimal("25.00"),
            moq=6,
            source="synthetic",
        ),
        InventoryParam(
            product_id=3,
            store_id=1,
            lead_time_days=Decimal("7.0"),
            lead_time_std_days=Decimal("1.40"),
            service_level=Decimal("0.950"),
            unit_cost=Decimal("10.0000"),
            holding_cost_rate=Decimal("0.2000"),
            order_cost=Decimal("25.00"),
            moq=6,
            source="synthetic",
        ),
        InventoryParam(
            product_id=4,
            store_id=1,
            lead_time_days=Decimal("14.0"),
            lead_time_std_days=Decimal("2.80"),
            service_level=Decimal("0.900"),
            unit_cost=Decimal("5.0000"),
            holding_cost_rate=Decimal("0.2000"),
            order_cost=Decimal("25.00"),
            moq=12,
            source="synthetic",
        ),
    ]
    session.add_all(params)

    # 7. Inventory snapshots at cutoff date
    snapshots = [
        InventorySnapshot(
            snapshot_date=cutoff,
            store_id=1,
            product_id=1,
            on_hand=10,
            on_order=0,
            source="synthetic",
        ),
        InventorySnapshot(
            snapshot_date=cutoff,
            store_id=1,
            product_id=2,
            on_hand=600,
            on_order=0,
            source="synthetic",
        ),
        InventorySnapshot(
            snapshot_date=cutoff,
            store_id=1,
            product_id=3,
            on_hand=150,
            on_order=0,
            source="synthetic",
        ),
        InventorySnapshot(
            snapshot_date=cutoff,
            store_id=1,
            product_id=4,
            on_hand=0,
            on_order=0,
            source="synthetic",
        ),
    ]
    session.add_all(snapshots)

    # 8. Forecast rows (Products 1, 2, 3 have 7-day forecasts; Product 4 has none)
    forecast_rows: list[Forecast] = []
    for h in range(1, 8):
        target = cutoff + datetime.timedelta(days=h)
        # Product 1: 10 units/day
        forecast_rows.append(
            Forecast(
                model_run_id=forecast_run_id,
                store_id=1,
                product_id=1,
                origin_date=cutoff,
                target_date=target,
                horizon_day=h,
                yhat=Decimal("10.0000"),
                yhat_lower=Decimal("8.0000"),
                yhat_upper=Decimal("12.0000"),
                interval_level=Decimal("0.80"),
            )
        )
        # Product 2: 5 units/day
        forecast_rows.append(
            Forecast(
                model_run_id=forecast_run_id,
                store_id=1,
                product_id=2,
                origin_date=cutoff,
                target_date=target,
                horizon_day=h,
                yhat=Decimal("5.0000"),
                yhat_lower=Decimal("3.0000"),
                yhat_upper=Decimal("7.0000"),
                interval_level=Decimal("0.80"),
            )
        )
        # Product 3: 5 units/day
        forecast_rows.append(
            Forecast(
                model_run_id=forecast_run_id,
                store_id=1,
                product_id=3,
                origin_date=cutoff,
                target_date=target,
                horizon_day=h,
                yhat=Decimal("5.0000"),
                yhat_lower=Decimal("3.0000"),
                yhat_upper=Decimal("7.0000"),
                interval_level=Decimal("0.80"),
            )
        )

    session.add_all(forecast_rows)
    session.commit()

    return session


def test_generate_recommendations_all_four_statuses(
    in_memory_recommend_db: Session,
) -> None:
    """Verify inventory recommendations accurately cover all four distinct statuses."""
    session = in_memory_recommend_db
    cutoff = datetime.date(2011, 12, 8)

    res = generate_inventory_recommendations(session=session, store_id=1, data_as_of=cutoff)

    assert res.n_recommendations == 4
    assert res.active_output_updated is True

    # Check status counts contains all 4 statuses
    assert res.status_counts.get("reorder_now") == 1
    assert res.status_counts.get("overstock") == 1
    assert res.status_counts.get("ok") == 1
    assert res.status_counts.get("no_data") == 1

    # Check database records
    recs = (
        session.execute(
            sa.select(InventoryRecommendation).where(
                InventoryRecommendation.as_of_date == cutoff,
                InventoryRecommendation.store_id == 1,
            )
        )
        .scalars()
        .all()
    )
    assert len(recs) == 4
    by_pid = {r.product_id: r for r in recs}

    # 1. Product 1: reorder_now
    p1 = by_pid[1]
    assert p1.status == "reorder_now"
    assert p1.order_quantity > 0
    assert int(p1.order_quantity) % 6 == 0  # MOQ = 6
    assert p1.on_hand + p1.order_quantity >= p1.reorder_point
    assert float(p1.expected_stockout_prob) > 0.50
    assert p1.reorder_urgency > 50

    # 2. Product 2: overstock
    p2 = by_pid[2]
    assert p2.status == "overstock"
    assert p2.order_quantity == Decimal("0")
    assert p2.days_of_cover is not None
    assert float(p2.days_of_cover) > 90.0
    assert float(p2.expected_stockout_prob) < 0.05

    # 3. Product 3: ok
    p3 = by_pid[3]
    assert p3.status == "ok"
    assert p3.order_quantity == Decimal("0")
    assert p3.days_of_cover is not None
    assert float(p3.days_of_cover) <= 90.0

    # 4. Product 4: no_data
    p4 = by_pid[4]
    assert p4.status == "no_data"
    assert p4.order_quantity == Decimal("0")
    assert p4.days_of_cover is None
    assert p4.demand_mean_lt == Decimal("0")


def test_active_output_pointer_updated(in_memory_recommend_db: Session) -> None:
    """Verify ml.active_output correctly points to inventory recommendations."""
    session = in_memory_recommend_db
    cutoff = datetime.date(2011, 12, 8)

    res = generate_inventory_recommendations(session=session, store_id=1, data_as_of=cutoff)

    ptr = session.get(ActiveOutput, TASK_INVENTORY)
    assert ptr is not None
    assert ptr.model_run_id == res.forecast_run_id
    assert ptr.data_as_of == cutoff


def test_recommendations_idempotent(in_memory_recommend_db: Session) -> None:
    """Verify running recommendation generation repeatedly cleanly overwrites prior rows."""
    session = in_memory_recommend_db
    cutoff = datetime.date(2011, 12, 8)

    res1 = generate_inventory_recommendations(session=session, store_id=1, data_as_of=cutoff)
    assert res1.n_recommendations == 4

    res2 = generate_inventory_recommendations(session=session, store_id=1, data_as_of=cutoff)
    assert res2.n_recommendations == 4

    total_count = session.execute(
        sa.select(sa.func.count(InventoryRecommendation.recommendation_id))
    ).scalar()
    assert total_count == 4


def test_missing_forecast_run_raises(in_memory_recommend_db: Session) -> None:
    """Verify missing forecast run raises informative ValueError."""
    session = in_memory_recommend_db
    fake_id = uuid.uuid4()
    with pytest.raises(ValueError, match="not found"):
        generate_inventory_recommendations(session=session, forecast_run_id=fake_id)


def test_cli_score_inventory(in_memory_recommend_db: Session, monkeypatch: Any) -> None:
    """Verify retailpulse score inventory CLI works in JSON and formatted modes."""
    session = in_memory_recommend_db

    @contextmanager
    def mock_session_scope():  # type: ignore[no-untyped-def]
        yield session

    monkeypatch.setattr("retailpulse.cli.session_scope", mock_session_scope)

    # 1. JSON mode
    res_json = runner.invoke(
        app,
        ["score", "inventory", "--store-id", "1", "--data-as-of", "2011-12-08", "--json"],
    )
    assert res_json.exit_code == 0
    assert '"status": "succeeded"' in res_json.output
    assert '"n_recommendations": 4' in res_json.output

    # 2. Text mode
    res_text = runner.invoke(
        app,
        ["score", "inventory", "--store-id", "1", "--data-as-of", "2011-12-08"],
    )
    assert res_text.exit_code == 0
    assert "Inventory Recommendations Generated" in res_text.output
    assert "Active Output Pointer: UPDATED" in res_text.output
    assert "REORDER_NOW" in res_text.output
    assert "OVERSTOCK" in res_text.output

    # 3. Invalid date
    res_err = runner.invoke(app, ["score", "inventory", "--data-as-of", "not-a-date"])
    assert res_err.exit_code == 1
    assert "Invalid date format" in res_err.output
