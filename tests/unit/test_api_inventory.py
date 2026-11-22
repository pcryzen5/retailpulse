"""Unit and integration tests for Inventory API endpoints.

Implements P5-06 verification:
- GET /api/v1/inventory/summary: Status counts, 10-bin histogram, synthetic params share (Viewer+).
- GET /api/v1/inventory/recommendations: Paginated & filtered list (Viewer+).
- GET /api/v1/inventory/recommendations/export: Formula-safe CSV export
  (Analyst+ only, 403 for Viewer).
- GET /api/v1/inventory/params/{product_id}: View product policy parameters (Viewer+).
- PUT /api/v1/inventory/params/{product_id}: Upsert policy parameters, source='user' (Analyst+).
- POST /api/v1/inventory/what-if: Recompute policy without database persistence (Analyst+).
  *Crucial verify:* What-if never modifies or adds rows to the database.
- POST /api/v1/inventory/simulate: Synchronous simulation with max 200 limit (Analyst+).
  *Verify:* limit > 200 returns 422 Problem Details.
- GET /api/v1/inventory/simulations/latest: Latest stored simulation run (Viewer+).
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
from retailpulse.db.models.core import DimDate, DimProduct, DimStore, FactDailySales, InventoryParam
from retailpulse.db.models.ml import (
    ActiveOutput,
    ForecastMetric,
    InventoryRecommendation,
    ModelRun,
)
from retailpulse.db.session import get_db
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def inv_db_engine():
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
def inv_db_session(inv_db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=inv_db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_inventory_db(inv_db_session: Session) -> dict[str, Any]:
    """Seed test users, stores, products, inventory params, and recommendations."""
    as_of = datetime.date(2026, 11, 20)

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
    inv_db_session.add_all([viewer, analyst])

    # 2. Store & Products
    store = DimStore(
        store_id=1,
        store_code="WEB-01",
        store_name="Flagship Web Store",
        country="United Kingdom",
    )
    prod1 = DimProduct(
        product_id=1,
        sku="85123A",
        description="WHITE HANGING HEART T-LIGHT HOLDER",
        abc_class="A",
        xyz_class="X",
        is_product=True,
        unit_cost_est=Decimal("1.5000"),
    )
    prod2 = DimProduct(
        product_id=2,
        sku="71053",
        description="WHITE METAL LANTERN",
        abc_class="B",
        xyz_class="Y",
        is_product=True,
        unit_cost_est=Decimal("2.5000"),
    )
    prod3 = DimProduct(
        product_id=3,
        sku="84406B",
        description="CREAM CUPID HEARTS COAT HANGER",
        abc_class="C",
        xyz_class="Z",
        is_product=True,
        unit_cost_est=Decimal("3.0000"),
    )
    # Product with formula injection risk in description
    prod_formula = DimProduct(
        product_id=4,
        sku="99999F",
        description="=cmd|' /C calc'!A0",
        abc_class="A",
        xyz_class="X",
        is_product=True,
        unit_cost_est=Decimal("5.0000"),
    )
    inv_db_session.add_all([store, prod1, prod2, prod3, prod_formula])

    # 3. Model run and active outputs
    model_run = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="lightgbm",
        mlflow_run_id="mlflow_fc_1",
        status="succeeded",
        is_champion=True,
        git_sha="abcdef1",
        dataset_hash="hash123",
        data_end=as_of,
        seed=42,
    )
    inv_db_session.add(model_run)

    active_inv = ActiveOutput(
        task="inventory",
        model_run_id=model_run.run_id,
        data_as_of=as_of,
    )
    inv_db_session.add(active_inv)

    # 4. Inventory Parameters
    param1 = InventoryParam(
        product_id=1,
        store_id=1,
        lead_time_days=Decimal("14.0"),
        lead_time_std_days=Decimal("2.8"),
        service_level=Decimal("0.950"),
        unit_cost=Decimal("1.5000"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        moq=12,
        source="synthetic",
    )
    param2 = InventoryParam(
        product_id=2,
        store_id=1,
        lead_time_days=Decimal("21.0"),
        lead_time_std_days=Decimal("4.2"),
        service_level=Decimal("0.900"),
        unit_cost=Decimal("2.5000"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        moq=1,
        source="user",
    )
    param_formula = InventoryParam(
        product_id=4,
        store_id=1,
        lead_time_days=Decimal("14.0"),
        lead_time_std_days=Decimal("2.8"),
        service_level=Decimal("0.950"),
        unit_cost=Decimal("5.0000"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        moq=1,
        source="synthetic",
    )
    inv_db_session.add_all([param1, param2, param_formula])

    # 5. Inventory Recommendations
    rec1 = InventoryRecommendation(
        recommendation_id=uuid.uuid4(),
        forecast_run_id=model_run.run_id,
        store_id=1,
        product_id=1,
        as_of_date=as_of,
        on_hand=10,
        on_order=0,
        lead_time_days=Decimal("14.0"),
        service_level=Decimal("0.950"),
        demand_mean_lt=Decimal("280.0"),
        demand_std_lt=Decimal("56.0"),
        reorder_point=Decimal("372.0"),
        order_quantity=Decimal("240"),
        safety_stock=Decimal("92.0"),
        status="reorder_now",
        reorder_urgency=95,
        expected_stockout_prob=Decimal("0.9500"),
        days_of_cover=Decimal("2.5"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        unit_cost=Decimal("1.5000"),
        params_source="synthetic",
    )
    rec2 = InventoryRecommendation(
        recommendation_id=uuid.uuid4(),
        forecast_run_id=model_run.run_id,
        store_id=1,
        product_id=2,
        as_of_date=as_of,
        on_hand=500,
        on_order=0,
        lead_time_days=Decimal("21.0"),
        service_level=Decimal("0.900"),
        demand_mean_lt=Decimal("105.0"),
        demand_std_lt=Decimal("25.0"),
        reorder_point=Decimal("137.0"),
        order_quantity=Decimal("100"),
        safety_stock=Decimal("32.0"),
        status="ok",
        reorder_urgency=20,
        expected_stockout_prob=Decimal("0.2000"),
        days_of_cover=Decimal("100.0"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        unit_cost=Decimal("2.5000"),
        params_source="user",
    )
    rec3 = InventoryRecommendation(
        recommendation_id=uuid.uuid4(),
        forecast_run_id=model_run.run_id,
        store_id=1,
        product_id=3,
        as_of_date=as_of,
        on_hand=1500,
        on_order=0,
        lead_time_days=Decimal("28.0"),
        service_level=Decimal("0.900"),
        demand_mean_lt=Decimal("70.0"),
        demand_std_lt=Decimal("15.0"),
        reorder_point=Decimal("89.0"),
        order_quantity=Decimal("50"),
        safety_stock=Decimal("19.0"),
        status="overstock",
        reorder_urgency=5,
        expected_stockout_prob=Decimal("0.0500"),
        days_of_cover=Decimal("600.0"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        unit_cost=Decimal("3.0000"),
        params_source="synthetic",
    )
    rec_formula = InventoryRecommendation(
        recommendation_id=uuid.uuid4(),
        forecast_run_id=model_run.run_id,
        store_id=1,
        product_id=4,
        as_of_date=as_of,
        on_hand=12,
        on_order=0,
        lead_time_days=Decimal("14.0"),
        service_level=Decimal("0.950"),
        demand_mean_lt=Decimal("100.0"),
        demand_std_lt=Decimal("20.0"),
        reorder_point=Decimal("133.0"),
        order_quantity=Decimal("50"),
        safety_stock=Decimal("33.0"),
        status="reorder_now",
        reorder_urgency=85,
        expected_stockout_prob=Decimal("0.8500"),
        days_of_cover=Decimal("5.0"),
        holding_cost_rate=Decimal("0.2000"),
        order_cost=Decimal("25.00"),
        unit_cost=Decimal("5.0000"),
        params_source="synthetic",
    )
    inv_db_session.add_all([rec1, rec2, rec3, rec_formula])

    # 6. Dates and Daily Sales for simulator
    dt1 = as_of - datetime.timedelta(days=1)
    dt2 = as_of
    d1 = DimDate(
        date_key=dt1,
        year=dt1.year,
        quarter=(dt1.month - 1) // 3 + 1,
        month=dt1.month,
        iso_week=dt1.isocalendar()[1],
        day_of_week=dt1.weekday(),
        day_of_month=dt1.day,
        is_weekend=dt1.weekday() >= 5,
        is_trading_day=True,
        is_complete=True,
    )
    d2 = DimDate(
        date_key=dt2,
        year=dt2.year,
        quarter=(dt2.month - 1) // 3 + 1,
        month=dt2.month,
        iso_week=dt2.isocalendar()[1],
        day_of_week=dt2.weekday(),
        day_of_month=dt2.day,
        is_weekend=dt2.weekday() >= 5,
        is_trading_day=True,
        is_complete=True,
    )
    inv_db_session.add_all([d1, d2])

    s1 = FactDailySales(
        date_key=dt1,
        product_id=1,
        store_id=1,
        units_sold=20,
        units_returned=0,
        revenue=Decimal("30.00"),
        n_orders=5,
    )
    s2 = FactDailySales(
        date_key=dt2,
        product_id=1,
        store_id=1,
        units_sold=25,
        units_returned=0,
        revenue=Decimal("37.50"),
        n_orders=6,
    )
    inv_db_session.add_all([s1, s2])

    # Forecast metric for simulate
    fm = ForecastMetric(
        model_run_id=model_run.run_id,
        scope="global",
        scope_key="all",
        metric="rmse",
        value=15.5,
    )
    inv_db_session.add(fm)

    inv_db_session.commit()

    return {
        "viewer_id": viewer.user_id,
        "analyst_id": analyst.user_id,
        "model_run_id": model_run.run_id,
        "as_of": as_of,
    }


@pytest.fixture
def client_and_tokens(inv_db_session: Session, seeded_inventory_db: dict[str, Any]):
    """Create test client and JWT tokens for viewer and analyst."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: inv_db_session

    client = TestClient(app)

    viewer_token = create_access_token(
        user_id=seeded_inventory_db["viewer_id"],
        role="viewer",
    )
    analyst_token = create_access_token(
        user_id=seeded_inventory_db["analyst_id"],
        role="analyst",
    )

    return {
        "client": client,
        "viewer_headers": {"Authorization": f"Bearer {viewer_token}"},
        "analyst_headers": {"Authorization": f"Bearer {analyst_token}"},
    }


# --- Tests ---


def test_get_inventory_summary(client_and_tokens):
    """Test GET /api/v1/inventory/summary metrics, counts, and histogram."""
    client = client_and_tokens["client"]
    headers = client_and_tokens["viewer_headers"]

    resp = client.get("/api/v1/inventory/summary", headers=headers)
    assert resp.status_code == 200
    body = resp.json()

    assert body["total_items"] == 4
    assert body["status_counts"]["reorder_now"] == 2
    assert body["status_counts"]["ok"] == 1
    assert body["status_counts"]["overstock"] == 1
    assert body["status_counts"]["stockout_risk"] == 0

    # 3 of 4 recs are synthetic
    assert body["params_synthetic_share"] == 0.75

    # 10 bins histogram
    histogram = body["stockout_risk_histogram"]
    assert len(histogram) == 10
    total_histogram_count = sum(b["count"] for b in histogram)
    assert total_histogram_count == 4


def test_get_inventory_recommendations_pagination_and_filters(client_and_tokens):
    """Test GET /api/v1/inventory/recommendations list, filters, and sort."""
    client = client_and_tokens["client"]
    headers = client_and_tokens["viewer_headers"]

    # 1. Default list
    resp = client.get("/api/v1/inventory/recommendations?page=1&page_size=2", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 4
    assert len(body["items"]) == 2
    assert body["pages"] == 2
    assert body["params_synthetic_share"] == 0.75

    # 2. Filter by status
    resp = client.get("/api/v1/inventory/recommendations?status=reorder_now", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    for item in body["items"]:
        assert item["status"] == "reorder_now"

    # 3. Filter by abc_class
    resp = client.get("/api/v1/inventory/recommendations?abc_class=B", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["sku"] == "71053"

    # 4. Search q
    resp = client.get("/api/v1/inventory/recommendations?q=LANTERN", headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["sku"] == "71053"

    # 5. Sort ascending by stockout_risk
    resp = client.get(
        "/api/v1/inventory/recommendations?sort=stockout_risk:asc",
        headers=headers,
    )
    assert resp.status_code == 200
    items = resp.json()["items"]
    risks = [item["stockout_risk"] for item in items]
    assert risks == sorted(risks)


def test_export_inventory_recommendations_rbac_and_formula_safety(client_and_tokens):
    """Test GET /api/v1/inventory/recommendations/export RBAC and formula neutralization."""
    client = client_and_tokens["client"]

    # 1. Viewer forbidden (403)
    resp = client.get(
        "/api/v1/inventory/recommendations/export",
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 403

    # 2. Analyst allowed (200)
    resp = client.get(
        "/api/v1/inventory/recommendations/export",
        headers=client_and_tokens["analyst_headers"],
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]
    assert "inventory_recommendations.csv" in resp.headers["content-disposition"]

    content = resp.text
    lines = [line.strip() for line in content.strip().splitlines()]
    assert len(lines) >= 5  # header + 4 rows
    assert "product_id,sku,description,abc_class" in lines[0]

    # SEC-INP-08: Ensure formula triggers (=, +, -, @) are sanitized with leading single quote
    formula_lines = [line for line in lines if "99999F" in line]
    assert len(formula_lines) == 1
    # '=cmd|' /C calc'!A0 must be sanitized as '\'=cmd|...'
    assert "'=cmd|" in formula_lines[0]


def test_inventory_params_get_and_put(client_and_tokens, inv_db_session: Session):
    """Test GET and PUT /api/v1/inventory/params/{product_id}."""
    client = client_and_tokens["client"]

    # 1. GET for existing product
    resp = client.get(
        "/api/v1/inventory/params/1?store_id=1",
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["product_id"] == 1
    assert body["lead_time_days"] == 14.0
    assert body["source"] == "synthetic"

    # 2. GET for non-existent product
    resp = client.get(
        "/api/v1/inventory/params/999?store_id=1",
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 404

    # 3. PUT with viewer role -> 403 Forbidden
    payload = {
        "lead_time_days": 10.0,
        "lead_time_std_days": 2.0,
        "service_level": 0.98,
        "unit_cost": 2.00,
        "holding_cost_rate": 0.25,
        "order_cost": 30.0,
        "moq": 24,
    }
    resp = client.put(
        "/api/v1/inventory/params/1?store_id=1",
        json=payload,
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 403

    # 4. PUT with analyst role -> 200 OK and source becomes 'user'
    resp = client.put(
        "/api/v1/inventory/params/1?store_id=1",
        json=payload,
        headers=client_and_tokens["analyst_headers"],
    )
    assert resp.status_code == 200
    updated_body = resp.json()
    assert updated_body["lead_time_days"] == 10.0
    assert updated_body["service_level"] == 0.98
    assert updated_body["source"] == "user"

    # Verify directly in DB
    param = inv_db_session.get(InventoryParam, (1, 1))
    assert param is not None
    assert param.source == "user"
    assert param.lead_time_days == Decimal("10.0")


def test_inventory_what_if_non_persistence(client_and_tokens, inv_db_session: Session):
    """Test POST /api/v1/inventory/what-if and verify no DB changes occur."""
    client = client_and_tokens["client"]

    # 1. Viewer forbidden
    resp = client.post(
        "/api/v1/inventory/what-if",
        json={"product_id": 1, "store_id": 1, "overrides": {"lead_time_days": 7}},
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 403

    # 2. Count DB records before what-if
    rec_count_before = (
        inv_db_session.scalar(sa.select(sa.func.count()).select_from(InventoryRecommendation)) or 0
    )
    param_count_before = (
        inv_db_session.scalar(sa.select(sa.func.count()).select_from(InventoryParam)) or 0
    )

    # 3. Analyst runs what-if with overrides
    what_if_payload = {
        "product_id": 1,
        "store_id": 1,
        "overrides": {
            "lead_time_days": 7.0,
            "service_level": 0.98,
            "moq": 12,
        },
    }
    resp = client.post(
        "/api/v1/inventory/what-if",
        json=what_if_payload,
        headers=client_and_tokens["analyst_headers"],
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["product_id"] == 1
    assert body["sku"] == "85123A"
    assert body["lead_time_days"] == 7.0
    assert body["service_level"] == 0.98
    assert body["persisted"] is False
    assert body["params_source"] == "what_if"

    # 4. CRITICAL: Verify DB record count is strictly unchanged!
    rec_count_after = (
        inv_db_session.scalar(sa.select(sa.func.count()).select_from(InventoryRecommendation)) or 0
    )
    param_count_after = (
        inv_db_session.scalar(sa.select(sa.func.count()).select_from(InventoryParam)) or 0
    )

    assert rec_count_after == rec_count_before
    assert param_count_after == param_count_before

    # 5. Invalid overrides return 422
    invalid_resp = client.post(
        "/api/v1/inventory/what-if",
        json={"product_id": 1, "store_id": 1, "overrides": {"lead_time_days": -5}},
        headers=client_and_tokens["analyst_headers"],
    )
    assert invalid_resp.status_code == 422


def test_inventory_simulate_limits_and_run(client_and_tokens):
    """Test POST /api/v1/inventory/simulate max 200 limit and GET latest."""
    client = client_and_tokens["client"]

    # 1. Latest before simulation -> 404
    resp = client.get(
        "/api/v1/inventory/simulations/latest",
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 404

    # 2. Exceed limit (201 > 200) -> 422 Problem Details
    resp = client.post(
        "/api/v1/inventory/simulate",
        json={"scope": {"limit": 201}},
        headers=client_and_tokens["analyst_headers"],
    )
    assert resp.status_code == 422
    assert "Simulation limit cannot exceed 200 SKUs" in resp.json()["detail"]

    # 3. Viewer forbidden from running simulate
    resp = client.post(
        "/api/v1/inventory/simulate",
        json={"scope": {"limit": 10}},
        headers=client_and_tokens["viewer_headers"],
    )
    assert resp.status_code == 403

    # 4. Valid simulation with Analyst
    sim_payload = {
        "scope": {"abc_class": "A", "limit": 10},
        "policy": {"service_level": 0.95},
        "baseline": {"type": "moving_average_28", "cover_days_extra": 14},
    }
    resp = client.post(
        "/api/v1/inventory/simulate",
        json=sim_payload,
        headers=client_and_tokens["analyst_headers"],
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "simulation_id" in body
    assert "window" in body
    assert "results" in body
    assert "retailpulse" in body["results"]
    assert "baseline" in body["results"]

    # 5. GET /api/v1/inventory/simulations/latest now returns 200
    latest_resp = client.get(
        "/api/v1/inventory/simulations/latest",
        headers=client_and_tokens["viewer_headers"],
    )
    assert latest_resp.status_code == 200
    latest_body = latest_resp.json()
    assert latest_body["simulation_id"] == body["simulation_id"]
    assert latest_body["window"] == body["window"]
