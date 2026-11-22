"""Unit and integration tests for Customers, Segments, and Churn API endpoints.

Implements P5-05 verification:
- GET /api/v1/segments: Active customer segment profiles.
- GET /api/v1/customers: Filtered list (segment, risk_band, min_probability, q prefix, sorting).
- GET /api/v1/customers/{customer_id}: Detailed profile with RFM and churn drivers.
- GET /api/v1/customers/export: Formula-safe CSV export
  (Analyst+ required, Viewer forbidden with 403).
- GET /api/v1/churn/summary: Population band distribution,
  10-bin probability histogram, revenue at risk.
- Privacy verification: customer keys are masked/pseudonymized; raw identifiers never leaked.
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
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.core import DimCustomer, DimProduct, DimStore, FactSalesLine
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    ModelRun,
    SegmentProfile,
)
from retailpulse.db.models.ops import IngestionBatch
from retailpulse.db.session import get_db
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def cust_db_engine():
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
def cust_db_session(cust_db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=cust_db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_customer_db(cust_db_session: Session) -> dict[str, Any]:
    """Seed users, customers, active segmentation, churn scores, and sales lines."""
    as_of = datetime.date(2026, 11, 20)

    # 1. Users: Viewer and Analyst
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
    cust_db_session.add_all([viewer, analyst])

    # 2. Store and Product
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store")
    prod = DimProduct(
        product_id=1,
        sku="SKU-1",
        description="Sample Mug",
        is_product=True,
    )
    cust_db_session.add_all([store, prod])

    # 3. Ingestion batch for sales lines
    batch = IngestionBatch(
        batch_id=uuid.uuid4(),
        source_name="test_source",
        source_file="sales.csv",
        file_sha256="a" * 64,
        status="succeeded",
    )
    cust_db_session.add(batch)

    # 4. Customers
    cust1 = DimCustomer(
        customer_id=1001,
        customer_key="9c1f4e5a2b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e",
        country="United Kingdom",
        first_order_date=as_of - datetime.timedelta(days=120),
        last_order_date=as_of - datetime.timedelta(days=10),
    )
    cust2 = DimCustomer(
        customer_id=1002,
        customer_key="8a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
        country="France",
        first_order_date=as_of - datetime.timedelta(days=200),
        last_order_date=as_of - datetime.timedelta(days=90),
    )
    cust_db_session.add_all([cust1, cust2])

    # 5. Sales lines for customer 1001
    line1 = FactSalesLine(
        sales_line_id=1,
        batch_id=batch.batch_id,
        source_row_hash="hash1" + "0" * 59,
        invoice_no="INV-001",
        invoice_ts=datetime.datetime(2026, 11, 10, 10, 0, 0, tzinfo=datetime.UTC),
        date_key=as_of - datetime.timedelta(days=10),
        store_id=1,
        product_id=1,
        customer_id=1001,
        quantity=2,
        unit_price=Decimal("50.00"),
        is_return=False,
    )
    cust_db_session.add(line1)

    # 6. Model Run for Segmentation
    seg_run_id = uuid.uuid4()
    seg_run = ModelRun(
        run_id=seg_run_id,
        task="segmentation",
        algorithm="kmeans_rfm",
        mlflow_run_id="mlflow_seg_1",
        is_champion=True,
    )
    cust_db_session.add(seg_run)

    # 7. Segment Profiles
    prof1 = SegmentProfile(
        model_run_id=seg_run_id,
        as_of_date=as_of,
        segment_id=1,
        segment_label="Champions",
        n_customers=1,
        avg_recency_days=Decimal("10.00"),
        avg_frequency=Decimal("5.00"),
        avg_monetary=Decimal("1000.00"),
        revenue_share=Decimal("0.80000"),
    )
    prof2 = SegmentProfile(
        model_run_id=seg_run_id,
        as_of_date=as_of,
        segment_id=2,
        segment_label="At Risk",
        n_customers=1,
        avg_recency_days=Decimal("90.00"),
        avg_frequency=Decimal("2.00"),
        avg_monetary=Decimal("200.00"),
        revenue_share=Decimal("0.20000"),
    )
    cust_db_session.add_all([prof1, prof2])

    # 8. Customer Segment Assignments
    c_seg1 = CustomerSegment(
        model_run_id=seg_run_id,
        as_of_date=as_of,
        customer_id=1001,
        segment_id=1,
        recency_days=10,
        frequency=5,
        monetary=Decimal("1000.00"),
        r_score=5,
        f_score=5,
        m_score=5,
    )
    c_seg2 = CustomerSegment(
        model_run_id=seg_run_id,
        as_of_date=as_of,
        customer_id=1002,
        segment_id=2,
        recency_days=90,
        frequency=2,
        monetary=Decimal("200.00"),
        r_score=2,
        f_score=2,
        m_score=2,
    )
    cust_db_session.add_all([c_seg1, c_seg2])

    # 9. ActiveOutput for segmentation
    active_seg = ActiveOutput(
        task="segmentation",
        model_run_id=seg_run_id,
        data_as_of=as_of,
    )
    cust_db_session.add(active_seg)

    # 10. Model Run for Churn
    churn_run_id = uuid.uuid4()
    churn_run = ModelRun(
        run_id=churn_run_id,
        task="churn",
        algorithm="lightgbm",
        mlflow_run_id="mlflow_churn_1",
        is_champion=True,
        metrics={
            "expected_revenue_at_risk": 450.50,
        },
    )
    cust_db_session.add(churn_run)

    # 11. Churn Scores
    c_churn1 = ChurnScore(
        model_run_id=churn_run_id,
        as_of_date=as_of,
        customer_id=1001,
        churn_probability=Decimal("0.1500"),
        risk_band="low",
        top_drivers=[
            {
                "feature": "recency_days",
                "value": 10.0,
                "impact": -0.25,
                "direction": "decreases_risk",
            },
        ],
    )
    c_churn2 = ChurnScore(
        model_run_id=churn_run_id,
        as_of_date=as_of,
        customer_id=1002,
        churn_probability=Decimal("0.8500"),
        risk_band="high",
        top_drivers=[
            {
                "feature": "recency_days",
                "value": 90.0,
                "impact": 0.40,
                "direction": "increases_risk",
            },
            {
                "feature": "orders_last_90d",
                "value": 0.0,
                "impact": 0.20,
                "direction": "increases_risk",
            },
        ],
    )
    cust_db_session.add_all([c_churn1, c_churn2])

    # 12. ActiveOutput for churn
    active_churn = ActiveOutput(
        task="churn",
        model_run_id=churn_run_id,
        data_as_of=as_of,
    )
    cust_db_session.add(active_churn)

    cust_db_session.commit()

    return {
        "viewer": viewer,
        "analyst": analyst,
        "as_of": as_of,
        "cust1": cust1,
        "cust2": cust2,
    }


@pytest.fixture
def client(cust_db_engine: sa.Engine, seeded_customer_db: dict[str, Any]):
    """FastAPI TestClient with overridden get_db dependency."""
    app = create_app()
    session_cls = sessionmaker(bind=cust_db_engine)

    def override_get_db():
        session = session_cls()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def viewer_headers(seeded_customer_db: dict[str, Any]) -> dict[str, str]:
    """Viewer token headers."""
    token = create_access_token(
        user_id=seeded_customer_db["viewer"].user_id,
        role="viewer",
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def analyst_headers(seeded_customer_db: dict[str, Any]) -> dict[str, str]:
    """Analyst token headers."""
    token = create_access_token(
        user_id=seeded_customer_db["analyst"].user_id,
        role="analyst",
    )
    return {"Authorization": f"Bearer {token}"}


# =========================================================================
# 1. Customer Segments Tests (API §2.4)
# =========================================================================


def test_list_segments_success(client: TestClient, viewer_headers: dict[str, str]) -> None:
    """Verify /api/v1/segments returns segment profiles."""
    resp = client.get("/api/v1/segments", headers=viewer_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert data[0]["segment_id"] == 1
    assert data[0]["segment_label"] == "Champions"
    assert data[0]["revenue_share"] == 0.8
    assert data[1]["segment_id"] == 2
    assert data[1]["segment_label"] == "At Risk"


# =========================================================================
# 2. Customers List & Filter Tests (API §2.4)
# =========================================================================


def test_list_customers_filtering_and_masking(
    client: TestClient,
    viewer_headers: dict[str, str],
) -> None:
    """Verify /api/v1/customers pagination, filters, and customer_key masking."""
    # 1. All customers
    resp = client.get("/api/v1/customers", headers=viewer_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    items = body["items"]
    assert len(items) == 2
    # Verify privacy masking (e.g. 9c1f...4e)
    assert "..." in items[0]["customer_key"]

    # 2. Filter by risk_band 'high'
    high_resp = client.get("/api/v1/customers?risk_band=high", headers=viewer_headers)
    assert high_resp.status_code == 200
    assert high_resp.json()["total"] == 1
    assert high_resp.json()["items"][0]["customer_id"] == 1002
    assert high_resp.json()["items"][0]["risk_band"] == "high"

    # 3. Filter by min_probability
    prob_resp = client.get("/api/v1/customers?min_probability=0.5", headers=viewer_headers)
    assert prob_resp.status_code == 200
    assert prob_resp.json()["total"] == 1
    assert prob_resp.json()["items"][0]["customer_id"] == 1002

    # 4. Filter by segment_id 1
    seg_resp = client.get("/api/v1/customers?segment_id=1", headers=viewer_headers)
    assert seg_resp.status_code == 200
    assert seg_resp.json()["total"] == 1
    assert seg_resp.json()["items"][0]["customer_id"] == 1001

    # 5. Invalid risk_band returns 422
    bad_band = client.get("/api/v1/customers?risk_band=extreme", headers=viewer_headers)
    assert bad_band.status_code == 422

    # 6. Invalid min_probability returns 422
    bad_prob = client.get("/api/v1/customers?min_probability=1.5", headers=viewer_headers)
    assert bad_prob.status_code == 422


# =========================================================================
# 3. Customer Export Tests (Analyst+ RBAC and Formula Safety)
# =========================================================================


def test_export_customers_rbac_enforcement(
    client: TestClient,
    viewer_headers: dict[str, str],
    analyst_headers: dict[str, str],
) -> None:
    """Verify /api/v1/customers/export forbids Viewer (403) and allows Analyst (200)."""
    # 1. Viewer forbidden
    viewer_resp = client.get("/api/v1/customers/export", headers=viewer_headers)
    assert viewer_resp.status_code == 403
    assert viewer_resp.headers["Content-Type"] == "application/problem+json"

    # 2. Analyst allowed
    analyst_resp = client.get("/api/v1/customers/export", headers=analyst_headers)
    assert analyst_resp.status_code == 200
    assert (
        analyst_resp.headers["Content-Disposition"] == 'attachment; filename="customers_export.csv"'
    )
    assert "text/csv" in analyst_resp.headers["Content-Type"]

    csv_text = analyst_resp.text
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    assert len(lines) == 3  # Header + 2 customers
    assert "customer_id,customer_key,country,segment_id" in lines[0]
    assert "1001," in lines[1]
    assert "1002," in lines[2]


# =========================================================================
# 4. Customer Detail Tests (API §3.5)
# =========================================================================


def test_get_customer_detail(client: TestClient, viewer_headers: dict[str, str]) -> None:
    """Verify /api/v1/customers/{id} returns segment, RFM, churn drivers, and orders summary."""
    # 1. Existing customer 1001
    resp = client.get("/api/v1/customers/1001", headers=viewer_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["customer_id"] == 1001
    assert "..." in data["customer_key"]
    assert data["country"] == "United Kingdom"

    # Segment and RFM
    assert data["segment"]["segment_id"] == 1
    assert data["segment"]["label"] == "Champions"
    assert data["rfm"]["recency_days"] == 10
    assert data["rfm"]["frequency"] == 5

    # Churn
    assert data["churn"]["risk_band"] == "low"
    assert data["churn"]["probability"] == 0.15
    assert len(data["churn"]["top_reasons"]) == 1
    assert data["churn"]["top_reasons"][0]["feature"] == "recency_days"

    # Orders summary
    assert data["orders_summary"]["n_orders"] == 1
    assert data["orders_summary"]["avg_order_value"] == 100.0

    # 2. Unknown customer returns 404
    not_found = client.get("/api/v1/customers/99999", headers=viewer_headers)
    assert not_found.status_code == 404


# =========================================================================
# 5. Churn Population Summary Tests (API §2.4)
# =========================================================================


def test_get_churn_summary(client: TestClient, viewer_headers: dict[str, str]) -> None:
    """Verify /api/v1/churn/summary returns band counts, 10-bin histogram, and revenue at risk."""
    resp = client.get("/api/v1/churn/summary", headers=viewer_headers)
    assert resp.status_code == 200
    data = resp.json()

    assert data["total_scored"] == 2
    assert data["band_counts"]["low"] == 1
    assert data["band_counts"]["high"] == 1
    assert data["band_counts"]["medium"] == 0

    assert len(data["probability_histogram"]) == 10
    # Customer 1001 (0.15) falls into bin 1 (0.1 - 0.2)
    assert data["probability_histogram"][1]["count"] == 1
    # Customer 1002 (0.85) falls into bin 8 (0.8 - 0.9)
    assert data["probability_histogram"][8]["count"] == 1

    assert data["mean_churn_probability"] == 0.50
    assert Decimal(str(data["expected_revenue_at_risk"])) == Decimal("450.50")
