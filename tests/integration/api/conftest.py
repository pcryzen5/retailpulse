"""Shared fixtures for API integration test suite.

Implements P5-10 testing infrastructure:
- Seeded database covering all core, ops, ml, and app tables.
- Bearer tokens for viewer, analyst, admin, and inactive accounts.
- TestClient with in-memory SQLite backend and disabled rate limits for clean integration runs.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.config import Settings
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.core import (
    DimCustomer,
    DimProduct,
    DimStore,
    InventoryParam,
)
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    InventoryRecommendation,
    ModelRun,
    SegmentProfile,
)
from retailpulse.db.models.ops import (
    Alert,
    DriftReport,
    IngestionBatch,
    PipelineRun,
)
from retailpulse.db.session import get_db
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password
from retailpulse.timeutil import utc_now


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def api_db_engine():
    """In-memory SQLite engine with translated schemas."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def api_db_session(api_db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=api_db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_api_data(api_db_session: Session) -> dict[str, Any]:
    """Seed comprehensive dataset across app, core, ops, and ml tables."""
    now = utc_now()
    today = now.date()

    # 1. Users
    viewer = AppUser(
        user_id=uuid.uuid4(),
        email="viewer@example.com",
        full_name="Test Viewer",
        role="viewer",
        password_hash=hash_password("ValidPassword123!"),
        is_active=True,
    )
    analyst = AppUser(
        user_id=uuid.uuid4(),
        email="analyst@example.com",
        full_name="Test Analyst",
        role="analyst",
        password_hash=hash_password("ValidPassword123!"),
        is_active=True,
    )
    admin = AppUser(
        user_id=uuid.uuid4(),
        email="admin@example.com",
        full_name="Test Admin",
        role="admin",
        password_hash=hash_password("ValidPassword123!"),
        is_active=True,
    )
    inactive_user = AppUser(
        user_id=uuid.uuid4(),
        email="inactive@example.com",
        full_name="Inactive User",
        role="viewer",
        password_hash=hash_password("ValidPassword123!"),
        is_active=False,
    )
    api_db_session.add_all([viewer, analyst, admin, inactive_user])

    # 2. Stores and Products
    store = DimStore(
        store_id=1,
        store_code="ONLINE",
        store_name="Online Store",
        country="United Kingdom",
    )
    product = DimProduct(
        product_id=101,
        sku="SKU101",
        description="EXAMPLE PRODUCT 101",
        category="General",
        abc_class="A",
        xyz_class="X",
        is_product=True,
        unit_cost_est=Decimal("12.50"),
        first_sale_date=today - datetime.timedelta(days=100),
        last_sale_date=today - datetime.timedelta(days=1),
    )
    customer = DimCustomer(
        customer_id=201,
        customer_key="9c1f000000000000000000000000000000000000000000000000000000000001",
        country="United Kingdom",
        first_order_date=today - datetime.timedelta(days=200),
        last_order_date=today - datetime.timedelta(days=10),
    )
    api_db_session.add_all([store, product, customer])

    # 3. Model Run and Active Output
    model_champ = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="lightgbm",
        mlflow_run_id="mlflow_champ_1",
        status="succeeded",
        is_champion=True,
        trained_at=now - datetime.timedelta(days=5),
        data_end=today - datetime.timedelta(days=5),
        metrics={"wape": 0.38, "gate_passed": True, "gate_reason": "Beats baseline"},
        params={"objective": "tweedie", "horizon": 28},
        git_sha="gitsha001",
        dataset_hash="hash001",
        seed=42,
    )
    api_db_session.add(model_champ)
    api_db_session.flush()

    active_out = ActiveOutput(
        task="forecast",
        model_run_id=model_champ.run_id,
        data_as_of=today - datetime.timedelta(days=1),
        scored_at=now - datetime.timedelta(days=4),
    )
    active_seg = ActiveOutput(
        task="segmentation",
        model_run_id=model_champ.run_id,
        data_as_of=today - datetime.timedelta(days=1),
        scored_at=now - datetime.timedelta(days=4),
    )
    active_churn = ActiveOutput(
        task="churn",
        model_run_id=model_champ.run_id,
        data_as_of=today - datetime.timedelta(days=1),
        scored_at=now - datetime.timedelta(days=4),
    )
    active_inv = ActiveOutput(
        task="inventory",
        model_run_id=model_champ.run_id,
        data_as_of=today - datetime.timedelta(days=1),
        scored_at=now - datetime.timedelta(days=4),
    )
    api_db_session.add_all([active_out, active_seg, active_churn, active_inv])

    seg_profile = SegmentProfile(
        model_run_id=model_champ.run_id,
        as_of_date=today - datetime.timedelta(days=1),
        segment_id=1,
        segment_label="Champions",
        n_customers=10,
        avg_recency_days=Decimal("15.00"),
        avg_frequency=Decimal("5.00"),
        avg_monetary=Decimal("500.00"),
        revenue_share=Decimal("0.50000"),
    )
    cust_seg = CustomerSegment(
        model_run_id=model_champ.run_id,
        as_of_date=today - datetime.timedelta(days=1),
        customer_id=customer.customer_id,
        segment_id=1,
        recency_days=10,
        frequency=5,
        monetary=Decimal("540.00"),
    )
    churn_score = ChurnScore(
        model_run_id=model_champ.run_id,
        as_of_date=today - datetime.timedelta(days=1),
        customer_id=customer.customer_id,
        churn_probability=Decimal("0.1500"),
        risk_band="low",
        top_drivers=[],
    )
    inv_rec = InventoryRecommendation(
        forecast_run_id=model_champ.run_id,
        as_of_date=today - datetime.timedelta(days=1),
        store_id=store.store_id,
        product_id=product.product_id,
        on_hand=50,
        on_order=0,
        lead_time_days=Decimal("14.0"),
        service_level=Decimal("0.950"),
        reorder_point=Decimal("20.00"),
        order_quantity=Decimal("30.00"),
        safety_stock=Decimal("10.00"),
        status="ok",
        reorder_urgency=20,
        expected_stockout_prob=Decimal("0.0500"),
        days_of_cover=Decimal("25.00"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        unit_cost=Decimal("12.5000"),
        params_source="user",
    )
    api_db_session.add_all([seg_profile, cust_seg, churn_score, inv_rec])

    # 4. Inventory Parameters
    inv_param = InventoryParam(
        product_id=product.product_id,
        store_id=store.store_id,
        lead_time_days=Decimal("14.0"),
        lead_time_std_days=Decimal("2.5"),
        service_level=Decimal("0.95"),
        unit_cost=Decimal("12.50"),
        holding_cost_rate=Decimal("0.20"),
        order_cost=Decimal("25.00"),
        moq=1,
        source="user",
        updated_at=now,
    )
    api_db_session.add(inv_param)

    # 5. Pipeline Run and Ingestion Batch
    batch = IngestionBatch(
        batch_id=uuid.uuid4(),
        source_name="generic_csv",
        source_file="uploads/sample.csv",
        original_filename="sample.csv",
        file_sha256="b" * 64,
        status="succeeded",
        row_count_in=500,
        row_count_loaded=500,
        row_count_rejected=0,
        created_at=now - datetime.timedelta(hours=2),
    )
    pipe_run = PipelineRun(
        run_id=uuid.uuid4(),
        flow_name="score_all",
        trigger="manual",
        status="succeeded",
        params={"scope": "all"},
        queued_at=now - datetime.timedelta(hours=1),
    )
    api_db_session.add_all([batch, pipe_run])

    # 6. Alert and Drift Report
    alert = Alert(
        severity="info",
        kind="drift_detected",
        message="5 features shifted",
        payload={"count": 5},
        dedupe_key="drift_test_001",
        created_at=now - datetime.timedelta(hours=3),
    )
    drift = DriftReport(
        report_id=uuid.uuid4(),
        task="forecast",
        model_run_id=model_champ.run_id,
        kind="data",
        reference_start=today - datetime.timedelta(days=60),
        reference_end=today - datetime.timedelta(days=30),
        current_start=today - datetime.timedelta(days=29),
        current_end=today,
        dataset_drift=False,
        share_drifted_features=Decimal("0.0"),
        summary={"drift": False},
        report_path="drift/sample.html",
        created_at=now,
    )
    api_db_session.add_all([alert, drift])

    api_db_session.commit()

    return {
        "viewer_id": viewer.user_id,
        "analyst_id": analyst.user_id,
        "admin_id": admin.user_id,
        "inactive_user_id": inactive_user.user_id,
        "product_id": product.product_id,
        "store_id": store.store_id,
        "customer_id": customer.customer_id,
        "model_run_id": model_champ.run_id,
        "batch_id": batch.batch_id,
        "pipeline_run_id": pipe_run.run_id,
        "alert_id": alert.alert_id,
        "drift_report_id": drift.report_id,
    }


@pytest.fixture
def api_client(api_db_session: Session, seeded_api_data: dict[str, Any]):
    """TestClient wired with in-memory DB and authenticated request headers."""
    settings = Settings(env="test")
    app = create_app(settings=settings)
    app.dependency_overrides[get_db] = lambda: api_db_session

    prev_limiter_enabled = True
    if hasattr(app.state, "limiter"):
        prev_limiter_enabled = getattr(app.state.limiter, "enabled", True)
        app.state.limiter.enabled = False

    client = TestClient(app)

    viewer_tok = create_access_token(
        user_id=seeded_api_data["viewer_id"],
        role="viewer",
        secret_key=settings.jwt_secret.get_secret_value(),
    )
    analyst_tok = create_access_token(
        user_id=seeded_api_data["analyst_id"],
        role="analyst",
        secret_key=settings.jwt_secret.get_secret_value(),
    )
    admin_tok = create_access_token(
        user_id=seeded_api_data["admin_id"],
        role="admin",
        secret_key=settings.jwt_secret.get_secret_value(),
    )

    try:
        yield {
            "client": client,
            "app": app,
            "headers": {
                "anonymous": {},
                "viewer": {"Authorization": f"Bearer {viewer_tok}"},
                "analyst": {"Authorization": f"Bearer {analyst_tok}"},
                "admin": {"Authorization": f"Bearer {admin_tok}"},
            },
            "seeds": seeded_api_data,
        }
    finally:
        if hasattr(app.state, "limiter"):
            app.state.limiter.enabled = prev_limiter_enabled
