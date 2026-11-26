"""Unit tests for retention policies and automatic champion promotion.

Verifies P7-08:
- Retention job per DATABASE §8:
  * Keeps latest 14 scoring dates per task in prediction tables.
  * Preserves whatever `ml.active_output` points to even when older than 14 dates.
  * Prunes audit logs older than 365 days.
  * Prunes drift reports older than 180 days and deletes their HTML reports from disk.
  * Deletes raw uploaded files 30 days after successful load while preserving batch metadata.
- `RP_AUTO_PROMOTE` (default false):
  * Default setting `RP_AUTO_PROMOTE=false` leaves champion unchanged.
  * When `RP_AUTO_PROMOTE=true`, passing candidate is promoted and audit-logged as `promote (auto)`.
- CLI commands (`retailpulse db prune`) execute end-to-end.
"""

from __future__ import annotations

import datetime
import sqlite3
import uuid
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

import retailpulse.ml.registry as reg
from retailpulse.cli import app
from retailpulse.config import Settings
from retailpulse.db.base import Base
from retailpulse.db.models.core import DimCustomer, DimProduct, DimStore
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    Forecast,
    InventoryRecommendation,
    ModelRun,
    SegmentProfile,
)
from retailpulse.db.models.ops import AuditLog, DriftReport, IngestionBatch
from retailpulse.db.retention import (
    prune_ops_retention,
    prune_prediction_retention,
    prune_raw_uploads_retention,
)
from retailpulse.ml.registry import promote_champion

sqlite3.register_adapter(Decimal, float)

runner = CliRunner()


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):  # type: ignore[no-untyped-def]
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def mock_db_session(monkeypatch: pytest.MonkeyPatch) -> Session:
    """In-memory SQLite session with table schema for retention testing."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    @contextmanager
    def _test_session_scope():
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise

    monkeypatch.setattr("retailpulse.db.session.session_scope", _test_session_scope)
    monkeypatch.setattr("retailpulse.pipelines.flows.session_scope", _test_session_scope)
    monkeypatch.setattr("retailpulse.cli.session_scope", _test_session_scope)
    return session


@pytest.fixture
def seed_core_dims(mock_db_session: Session) -> tuple[int, int, int]:
    """Seed parent dimension rows for foreign keys."""
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store")
    prod = DimProduct(
        product_id=101,
        sku="SKU-101",
        description="Test Product",
    )
    cust = DimCustomer(
        customer_id=1001,
        customer_key="CUST-1001",
    )
    mock_db_session.add_all([store, prod, cust])
    mock_db_session.commit()
    return store.store_id, prod.product_id, cust.customer_id


def test_rp_auto_promote_default_is_false() -> None:
    """Verify that auto_promote defaults to False per TRD §6 and PRD FR-OPS-06."""
    settings = Settings()
    assert settings.auto_promote is False


def test_prune_prediction_retention_keeps_active_pointer_rows(
    mock_db_session: Session,
    seed_core_dims: tuple[int, int, int],
) -> None:
    """Core verification: pruning keeps the latest 14 dates PLUS active pointer's older rows."""
    store_id, prod_id, _ = seed_core_dims

    # Create two model runs: active champion (old) and new runner
    old_run = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="Prophet",
        status="succeeded",
        is_champion=True,
        mlflow_run_id="mlflow-run-old",
    )
    new_run = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="LightGBM",
        status="succeeded",
        is_champion=False,
        mlflow_run_id="mlflow-run-new",
    )
    mock_db_session.add_all([old_run, new_run])
    mock_db_session.commit()

    # Old date: 2011-01-01 (far outside latest 14 dates)
    active_date = datetime.date(2011, 1, 1)

    # Active pointer explicitly points to old date & old run
    active_ptr = ActiveOutput(
        task="forecast",
        model_run_id=old_run.run_id,
        data_as_of=active_date,
    )
    mock_db_session.add(active_ptr)

    # Insert forecast for active pointer
    active_fc = Forecast(
        model_run_id=old_run.run_id,
        store_id=store_id,
        product_id=prod_id,
        origin_date=active_date,
        target_date=active_date + datetime.timedelta(days=1),
        horizon_day=1,
        yhat=Decimal("15.0000"),
        yhat_lower=Decimal("10.0000"),
        yhat_upper=Decimal("20.0000"),
    )
    mock_db_session.add(active_fc)

    # Insert 20 newer consecutive dates for new_run
    base_recent_date = datetime.date(2011, 10, 1)
    for i in range(20):
        fc_date = base_recent_date + datetime.timedelta(days=i)
        fc = Forecast(
            model_run_id=new_run.run_id,
            store_id=store_id,
            product_id=prod_id,
            origin_date=fc_date,
            target_date=fc_date + datetime.timedelta(days=1),
            horizon_day=1,
            yhat=Decimal("20.0000"),
            yhat_lower=Decimal("15.0000"),
            yhat_upper=Decimal("25.0000"),
        )
        mock_db_session.add(fc)

    mock_db_session.commit()

    total_before = mock_db_session.query(Forecast).count()
    assert total_before == 21  # 1 active old + 20 new

    # Execute pruning with keep_dates=14
    prune_res = prune_prediction_retention(mock_db_session, keep_dates=14)

    # Out of 20 new dates, the 6 oldest (2011-10-01 .. 2011-10-06) should be pruned
    assert prune_res["forecast"] == 6

    # Verify active pointer rows are still intact
    active_survivor = mock_db_session.execute(
        sa.select(Forecast).where(
            Forecast.model_run_id == old_run.run_id,
            Forecast.origin_date == active_date,
        )
    ).scalar_one_or_none()
    assert active_survivor is not None
    assert active_survivor.origin_date == active_date

    # Verify latest 14 dates survive
    surviving_dates = mock_db_session.execute(
        sa.select(Forecast.origin_date).distinct().order_by(Forecast.origin_date.desc())
    ).scalars().all()
    # 14 latest new dates + 1 active old date = 15 dates
    assert len(surviving_dates) == 15
    assert active_date in surviving_dates


def test_prune_prediction_retention_all_tasks(
    mock_db_session: Session,
    seed_core_dims: tuple[int, int, int],
) -> None:
    """Verify retention pruning across customer_segment, churn_score, and inventory."""
    store_id, prod_id, cust_id = seed_core_dims

    run_id = uuid.uuid4()
    run = ModelRun(
        run_id=run_id,
        task="segmentation",
        algorithm="KMeans",
        status="succeeded",
        mlflow_run_id="mlflow-run-seg",
    )
    mock_db_session.add(run)
    mock_db_session.commit()

    # Active pointer for segmentation on older date 2011-02-01
    active_seg_date = datetime.date(2011, 2, 1)
    mock_db_session.add(
        ActiveOutput(task="segmentation", model_run_id=run_id, data_as_of=active_seg_date)
    )

    # Active pointer for churn on older date 2011-03-01
    active_churn_date = datetime.date(2011, 3, 1)
    mock_db_session.add(
        ActiveOutput(task="churn", model_run_id=run_id, data_as_of=active_churn_date)
    )

    # Active pointer for inventory on older date 2011-04-01
    active_inv_date = datetime.date(2011, 4, 1)
    mock_db_session.add(
        ActiveOutput(task="inventory", model_run_id=run_id, data_as_of=active_inv_date)
    )

    # Add active rows
    mock_db_session.add(
        SegmentProfile(
            model_run_id=run_id,
            as_of_date=active_seg_date,
            segment_id=0,
            segment_label="Champions",
            n_customers=10,
            avg_recency_days=Decimal("5.00"),
            avg_frequency=Decimal("12.00"),
            avg_monetary=Decimal("500.00"),
            revenue_share=Decimal("0.50"),
        )
    )
    mock_db_session.add(
        CustomerSegment(
            model_run_id=run_id,
            as_of_date=active_seg_date,
            customer_id=cust_id,
            segment_id=0,
            recency_days=5,
            frequency=12,
            monetary=Decimal("500.00"),
        )
    )
    mock_db_session.add(
        ChurnScore(
            model_run_id=run_id,
            as_of_date=active_churn_date,
            customer_id=cust_id,
            churn_probability=Decimal("0.1234"),
            risk_band="low",
            top_drivers=[],
        )
    )
    def _make_inv(as_of: datetime.date) -> InventoryRecommendation:
        return InventoryRecommendation(
            forecast_run_id=run_id,
            as_of_date=as_of,
            store_id=store_id,
            product_id=prod_id,
            on_hand=100,
            lead_time_days=Decimal("7.0"),
            service_level=Decimal("0.950"),
            reorder_point=Decimal("50.00"),
            order_quantity=Decimal("100.00"),
            safety_stock=Decimal("20.00"),
            status="ok",
            reorder_urgency=10,
            expected_stockout_prob=Decimal("0.0500"),
            holding_cost_rate=Decimal("0.2000"),
            order_cost=Decimal("15.00"),
            unit_cost=Decimal("5.5000"),
        )

    mock_db_session.add(_make_inv(active_inv_date))

    # Insert 18 newer dates for each
    start_d = datetime.date(2011, 10, 1)
    for i in range(18):
        cur_d = start_d + datetime.timedelta(days=i)
        mock_db_session.add(
            SegmentProfile(
                model_run_id=run_id,
                as_of_date=cur_d,
                segment_id=0,
                segment_label="Champions",
                n_customers=10,
                avg_recency_days=Decimal("5.00"),
                avg_frequency=Decimal("12.00"),
                avg_monetary=Decimal("500.00"),
                revenue_share=Decimal("0.50"),
            )
        )
        mock_db_session.add(
            CustomerSegment(
                model_run_id=run_id,
                as_of_date=cur_d,
                customer_id=cust_id,
                segment_id=0,
                recency_days=5,
                frequency=12,
                monetary=Decimal("500.00"),
            )
        )
        mock_db_session.add(
            ChurnScore(
                model_run_id=run_id,
                as_of_date=cur_d,
                customer_id=cust_id,
                churn_probability=Decimal("0.2500"),
                risk_band="low",
                top_drivers=[],
            )
        )
        mock_db_session.add(_make_inv(cur_d))

    mock_db_session.commit()

    prune_res = prune_prediction_retention(mock_db_session, keep_dates=14)
    # 18 new dates - 14 kept = 4 pruned per task
    assert prune_res["customer_segment"] == 4
    assert prune_res["segment_profile"] == 4
    assert prune_res["churn_score"] == 4
    assert prune_res["inventory_recommendation"] == 4

    # Verify active rows remain
    assert (
        mock_db_session.execute(
            sa.select(CustomerSegment).where(CustomerSegment.as_of_date == active_seg_date)
        ).scalar_one_or_none()
        is not None
    )
    assert (
        mock_db_session.execute(
            sa.select(ChurnScore).where(ChurnScore.as_of_date == active_churn_date)
        ).scalar_one_or_none()
        is not None
    )
    assert (
        mock_db_session.execute(
            sa.select(InventoryRecommendation).where(
                InventoryRecommendation.as_of_date == active_inv_date
            )
        ).scalar_one_or_none()
        is not None
    )


def test_prune_ops_retention_unlinks_html_files(
    mock_db_session: Session,
    tmp_path: Path,
) -> None:
    """prune_ops_retention deletes drift reports > 180d, unlinks HTML, and prunes audit > 365d."""
    now = datetime.datetime(2026, 11, 26, 12, 0, 0, tzinfo=datetime.UTC)

    # Old drift report: 200 days ago
    old_drift_html = tmp_path / "old_drift.html"
    old_drift_html.write_text("<html>Old Drift Report</html>", encoding="utf-8")
    old_drift = DriftReport(
        report_id=uuid.uuid4(),
        task="forecast",
        kind="data",
        reference_start=datetime.date(2025, 1, 1),
        reference_end=datetime.date(2025, 2, 1),
        current_start=datetime.date(2025, 2, 2),
        current_end=datetime.date(2025, 3, 1),
        dataset_drift=True,
        report_path=str(old_drift_html),
        created_at=now - datetime.timedelta(days=200),
    )

    # Recent drift report: 10 days ago
    recent_drift_html = tmp_path / "recent_drift.html"
    recent_drift_html.write_text("<html>Recent Drift Report</html>", encoding="utf-8")
    recent_drift = DriftReport(
        report_id=uuid.uuid4(),
        task="forecast",
        kind="data",
        reference_start=datetime.date(2026, 11, 1),
        reference_end=datetime.date(2026, 11, 10),
        current_start=datetime.date(2026, 11, 11),
        current_end=datetime.date(2026, 11, 20),
        dataset_drift=False,
        report_path=str(recent_drift_html),
        created_at=now - datetime.timedelta(days=10),
    )

    # Old audit log: 400 days ago
    old_audit = AuditLog(
        ts=now - datetime.timedelta(days=400),
        action="user_login",
        resource="app:login",
    )
    # Recent audit log: 50 days ago
    recent_audit = AuditLog(
        ts=now - datetime.timedelta(days=50),
        action="promote",
        resource="model:forecast",
    )

    mock_db_session.add_all([old_drift, recent_drift, old_audit, recent_audit])
    mock_db_session.commit()

    ops_res = prune_ops_retention(
        mock_db_session,
        audit_retention_days=365,
        drift_retention_days=180,
        as_of=now,
    )

    assert ops_res["drift_report"] == 1
    assert ops_res["drift_files"] == 1
    assert ops_res["audit_log"] == 1

    # Old file must be unlinked, recent file kept
    assert not old_drift_html.exists()
    assert recent_drift_html.exists()

    # Old audit deleted, recent audit kept
    assert mock_db_session.query(AuditLog).count() == 1


def test_prune_raw_uploads_retention(mock_db_session: Session, tmp_path: Path) -> None:
    """prune_raw_uploads_retention deletes files > 30 days while preserving batch records."""
    now = datetime.datetime(2026, 11, 26, 12, 0, 0, tzinfo=datetime.UTC)

    old_file = tmp_path / "old_sales.csv"
    old_file.write_text("col1,col2\n1,2", encoding="utf-8")
    old_batch = IngestionBatch(
        batch_id=uuid.uuid4(),
        source_name="generic_csv",
        source_file=str(old_file),
        file_sha256="a" * 64,
        status="succeeded",
        created_at=now - datetime.timedelta(days=35),
    )

    recent_file = tmp_path / "recent_sales.csv"
    recent_file.write_text("col1,col2\n3,4", encoding="utf-8")
    recent_batch = IngestionBatch(
        batch_id=uuid.uuid4(),
        source_name="generic_csv",
        source_file=str(recent_file),
        file_sha256="b" * 64,
        status="succeeded",
        created_at=now - datetime.timedelta(days=5),
    )

    mock_db_session.add_all([old_batch, recent_batch])
    mock_db_session.commit()

    deleted = prune_raw_uploads_retention(mock_db_session, upload_retention_days=30, as_of=now)
    assert deleted == 1

    # Old file unlinked on disk, recent file preserved
    assert not old_file.exists()
    assert recent_file.exists()

    # Both batch rows remain in database
    assert mock_db_session.query(IngestionBatch).count() == 2


def test_auto_promote_flips_champion_and_audit_logs(mock_db_session: Session) -> None:
    """When auto_promote=True, promote_champion flips is_champion and logs 'promote (auto)'."""
    champ_run = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="Prophet",
        status="succeeded",
        is_champion=True,
        mlflow_run_id="mlflow-run-001",
    )
    new_candidate = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="LightGBM",
        status="succeeded",
        is_champion=False,
        mlflow_run_id="mlflow-run-002",
    )
    mock_db_session.add_all([champ_run, new_candidate])
    mock_db_session.commit()

    # Execute promote_champion with is_auto=True (mocking mlflow client)
    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.run_id = "mlflow-run-002"
    mock_mv.version = "2"
    mock_client.get_model_version.return_value = mock_mv
    mock_client.get_model_version_by_alias.return_value = MagicMock(version="1")

    orig_client = reg.MlflowClient
    reg.MlflowClient = MagicMock(return_value=mock_client)
    try:
        promote_champion(
            task="forecast",
            version="2",
            session=mock_db_session,
            is_auto=True,
        )
    finally:
        reg.MlflowClient = orig_client

    # Verify is_champion flipped
    mock_db_session.refresh(champ_run)
    mock_db_session.refresh(new_candidate)
    assert champ_run.is_champion is False
    assert new_candidate.is_champion is True

    # Verify audit log recorded "promote (auto)"
    audit = (
        mock_db_session.query(AuditLog).filter(AuditLog.action == "promote (auto)").one_or_none()
    )
    assert audit is not None
    assert f"model:forecast:{new_candidate.run_id}" in audit.resource
    assert audit.details.get("is_auto") is True


def test_cli_db_prune_command(mock_db_session: Session) -> None:
    """Test CLI command `retailpulse db prune --keep-dates 14 --json`."""
    res = runner.invoke(app, ["db", "prune", "--keep-dates", "14", "--json"])
    assert res.exit_code == 0
    assert '"total_rows_pruned": 0' in res.output
