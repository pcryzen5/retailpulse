"""Unit and CLI tests for inventory policy simulation report (P4-05).

Verifies:
  - Generation of reports/inventory_simulation.md with prominent synthetic-data caveat.
  - Generation of publication-ready diagnostic comparison figures.
  - Execution of CLI `retailpulse inventory simulate` in text and JSON modes.
  - Verification that report contains executive comparison and policy mechanics.
"""

from __future__ import annotations

import datetime
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
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
    FactDailySales,
    InventoryParam,
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    Forecast,
    ForecastMetric,
    ModelRun,
    PolicySimulation,
)
from retailpulse.ml.inventory.report import (
    generate_simulation_figures,
    generate_simulation_report,
)
from retailpulse.ml.inventory.simulate import (
    SimulationConfig,
    run_policy_simulation,
)

runner = CliRunner()


@pytest.fixture
def in_memory_report_db() -> Session:
    """Create in-memory SQLite database seeded with products, sales, params, and forecasts."""
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
        ModelRun.__table__,
        ForecastMetric.__table__,
        Forecast.__table__,
        PolicySimulation.__table__,
        ActiveOutput.__table__,
    ]:
        tbl.create(bind=engine, checkfirst=True)

    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    # 1. Store
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store", country="UK")
    session.add(store)

    # 2. Window dates: 28 simulation days + 28 historical days
    sim_end = datetime.date(2011, 12, 8)
    sim_start = datetime.date(2011, 11, 11)
    history_start = sim_start - datetime.timedelta(days=28)

    total_days = (sim_end - history_start).days + 1
    dim_dates = [
        DimDate(
            date_key=history_start + datetime.timedelta(days=i),
            year=2011,
            quarter=4,
            month=11,
            iso_week=45,
            day_of_week=1,
            day_of_month=15,
            is_weekend=False,
            is_holiday=False,
            is_trading_day=True,
            is_complete=True,
        )
        for i in range(total_days)
    ]
    session.add_all(dim_dates)

    # 3. Products
    products = [
        DimProduct(
            product_id=1,
            sku="SKU-RPT-A",
            description="Item Alpha",
            abc_class="A",
            unit_cost_est=Decimal("12.00"),
            is_product=True,
        ),
        DimProduct(
            product_id=2,
            sku="SKU-RPT-B",
            description="Item Beta",
            abc_class="B",
            unit_cost_est=Decimal("6.00"),
            is_product=True,
        ),
    ]
    session.add_all(products)

    # 4. Parameters
    params = [
        InventoryParam(
            store_id=1,
            product_id=1,
            lead_time_days=Decimal("7.0"),
            lead_time_std_days=Decimal("1.4"),
            service_level=Decimal("0.98"),
            holding_cost_rate=Decimal("0.20"),
            order_cost=Decimal("25.00"),
            moq=1,
            unit_cost=Decimal("12.00"),
            source="synthetic",
        ),
        InventoryParam(
            store_id=1,
            product_id=2,
            lead_time_days=Decimal("14.0"),
            lead_time_std_days=Decimal("2.8"),
            service_level=Decimal("0.95"),
            holding_cost_rate=Decimal("0.20"),
            order_cost=Decimal("25.00"),
            moq=6,
            unit_cost=Decimal("6.00"),
            source="synthetic",
        ),
    ]
    session.add_all(params)

    # 5. Sales
    sales = []
    for d in range(total_days):
        dt = history_start + datetime.timedelta(days=d)
        sales.append(
            FactDailySales(
                store_id=1,
                product_id=1,
                date_key=dt,
                units_sold=12,
                revenue=Decimal("180.00"),
                avg_unit_price=Decimal("15.00"),
                n_orders=6,
            )
        )
        sales.append(
            FactDailySales(
                store_id=1,
                product_id=2,
                date_key=dt,
                units_sold=4,
                revenue=Decimal("40.00"),
                avg_unit_price=Decimal("10.00"),
                n_orders=2,
            )
        )
    session.add_all(sales)

    # 6. ModelRun & Forecasts
    run_id = uuid.uuid4()
    model_run = ModelRun(
        run_id=run_id,
        task="forecast",
        algorithm="lgbm",
        mlflow_run_id="fake_mlflow_rpt",
        status="succeeded",
        is_champion=True,
        trained_at=datetime.datetime(2011, 11, 10, 12, 0, tzinfo=datetime.UTC),
    )
    session.add(model_run)

    metrics = [
        ForecastMetric(
            model_run_id=run_id,
            scope="abc",
            scope_key="A",
            metric="rmse",
            value=Decimal("1.20"),
        ),
        ForecastMetric(
            model_run_id=run_id,
            scope="abc",
            scope_key="B",
            metric="rmse",
            value=Decimal("1.80"),
        ),
        ForecastMetric(
            model_run_id=run_id,
            scope="global",
            scope_key="all",
            metric="rmse",
            value=Decimal("1.50"),
        ),
    ]
    session.add_all(metrics)

    forecast_rows = []
    cutoff_origin = datetime.date(2011, 11, 10)
    for d in range(28):
        dt = sim_start + datetime.timedelta(days=d)
        forecast_rows.append(
            Forecast(
                model_run_id=run_id,
                store_id=1,
                product_id=1,
                origin_date=cutoff_origin,
                target_date=dt,
                horizon_day=d + 1,
                yhat=Decimal("12.0"),
                yhat_lower=Decimal("9.0"),
                yhat_upper=Decimal("15.0"),
            )
        )
        forecast_rows.append(
            Forecast(
                model_run_id=run_id,
                store_id=1,
                product_id=2,
                origin_date=cutoff_origin,
                target_date=dt,
                horizon_day=d + 1,
                yhat=Decimal("4.0"),
                yhat_lower=Decimal("2.0"),
                yhat_upper=Decimal("6.0"),
            )
        )
    session.add_all(forecast_rows)

    session.commit()
    return session


def test_generate_simulation_report_file(in_memory_report_db: Session, tmp_path: Path) -> None:
    """Verify generate_simulation_report creates markdown file with prominent caveat."""
    session = in_memory_report_db
    config = SimulationConfig(
        store_id=1,
        window_start=datetime.date(2011, 11, 11),
        window_end=datetime.date(2011, 12, 8),
        seed=42,
        persist=False,
    )
    result = run_policy_simulation(session=session, config=config)

    report_path = tmp_path / "reports" / "inventory_simulation.md"
    fig_dir = tmp_path / "reports" / "figures"

    out_path = generate_simulation_report(
        result=result,
        output_path=report_path,
        figures_dir=fig_dir,
        session=session,
    )

    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")

    # 1. Prominent synthetic-data caveat
    assert "METHODOLOGICAL & DATA GOVERNANCE CAVEAT: SYNTHETIC INVENTORY PARAMETERS" in content
    assert "source='synthetic'" in content

    # 2. Executive summary table
    assert "Demand Fill Rate" in content
    assert "Stockout-Day Rate" in content
    assert "Average On-Hand Units" in content
    assert "Average Inventory Value" in content
    assert "Total Holding Cost" in content

    # 3. Policy formulation section
    assert "RetailPulse Forecast-Driven Policy" in content
    assert "Baseline Moving-Average Policy" in content

    # 4. Embedded figures
    assert (fig_dir / "inventory_fill_rate_comparison.png").exists()
    assert (fig_dir / "inventory_cost_comparison.png").exists()


def test_generate_simulation_figures(tmp_path: Path) -> None:
    """Verify figure generation produces non-empty PNG files."""
    results = {
        "retailpulse": {
            "fill_rate": 0.965,
            "stockout_day_rate": 0.035,
            "avg_on_hand_value": 25000.0,
            "holding_cost": 5000.0,
        },
        "baseline": {
            "fill_rate": 0.940,
            "stockout_day_rate": 0.060,
            "avg_on_hand_value": 28000.0,
            "holding_cost": 5600.0,
        },
    }
    fig_dir = tmp_path / "figures"
    figs = generate_simulation_figures(results, fig_dir)

    assert "fill_rate_comparison" in figs
    assert "cost_comparison" in figs
    assert figs["fill_rate_comparison"].stat().st_size > 0
    assert figs["cost_comparison"].stat().st_size > 0


def test_cli_inventory_simulate_json(in_memory_report_db: Session, monkeypatch: Any) -> None:
    """Verify retailpulse inventory simulate runs in JSON mode."""
    session = in_memory_report_db

    @contextmanager
    def mock_session_scope():  # type: ignore[no-untyped-def]
        yield session

    monkeypatch.setattr("retailpulse.cli.session_scope", mock_session_scope)

    res = runner.invoke(
        app,
        [
            "inventory",
            "simulate",
            "--store-id",
            "1",
            "--window-start",
            "2011-11-11",
            "--window-end",
            "2011-12-08",
            "--limit",
            "50",
            "--no-report",
            "--json",
        ],
    )

    assert res.exit_code == 0
    assert '"status": "succeeded"' in res.output
    assert '"simulation_id"' in res.output
    assert '"retailpulse"' in res.output
    assert '"baseline"' in res.output


def test_cli_inventory_simulate_formatted_report(
    in_memory_report_db: Session, monkeypatch: Any, tmp_path: Path
) -> None:
    """Verify retailpulse inventory simulate runs in text mode and writes report."""
    session = in_memory_report_db
    custom_report = tmp_path / "custom_sim_report.md"

    @contextmanager
    def mock_session_scope():  # type: ignore[no-untyped-def]
        yield session

    monkeypatch.setattr("retailpulse.cli.session_scope", mock_session_scope)

    res = runner.invoke(
        app,
        [
            "inventory",
            "simulate",
            "--store-id",
            "1",
            "--window-start",
            "2011-11-11",
            "--window-end",
            "2011-12-08",
            "--output",
            str(custom_report),
        ],
    )

    assert res.exit_code == 0
    assert "Inventory Policy Simulation Completed" in res.output
    assert "Demand Fill Rate:" in res.output
    assert custom_report.exists()
