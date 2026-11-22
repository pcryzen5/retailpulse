"""Unit and integration tests for Reference data, KPIs, and Forecast routers.

Implements P5-04 verification:
- Reference endpoints: /api/v1/stores, /api/v1/products, /api/v1/products/{id}, /api/v1/sales/daily.
- KPI overview: /api/v1/kpis/overview and /api/v1/kpis/summary, revenue 30d, caching.
- Forecasts: /api/v1/forecasts and /api/v1/forecasts/accuracy.
- 503 Service Unavailable before initial scoring.
- 404 unknown product, 422 ineligible product.
- Date range validation (<= 800 days).
- In-process 60-second caching and Cache-Control: no-store headers.
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
from retailpulse.api.cache import api_cache
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.core import DimCustomer, DimProduct, DimStore, FactDailySales
from retailpulse.db.models.ml import (
    ActiveOutput,
    Forecast,
    ForecastMetric,
    ModelRun,
)
from retailpulse.db.models.ops import PipelineRun
from retailpulse.db.session import get_db
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def ref_db_engine():
    """In-memory SQLite engine with translated schemas for unit testing."""
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
def ref_db_session(ref_db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=ref_db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_db(ref_db_session: Session) -> dict[str, Any]:
    """Seed sample reference data, daily sales, active model outputs, and users."""
    api_cache.clear()

    # 1. User
    viewer = AppUser(
        user_id=uuid.uuid4(),
        email="viewer@example.com",
        full_name="Viewer User",
        role="viewer",
        password_hash=hash_password("ValidPassword123!"),
        is_active=True,
    )
    ref_db_session.add(viewer)

    # 2. Store
    store = DimStore(
        store_id=1,
        store_code="ONLINE",
        store_name="Online Store",
        country="United Kingdom",
    )
    ref_db_session.add(store)

    # 3. Products: 1 eligible, 1 ineligible (short history), 1 non-merchandise
    today = datetime.date(2026, 11, 20)
    p_eligible = DimProduct(
        product_id=101,
        sku="SKU-ELIGIBLE-01",
        description="High Demand Ceramic Mug",
        category="Kitchen",
        is_product=True,
        unit_cost_est=Decimal("5.50"),
        abc_class="A",
        xyz_class="X",
        first_sale_date=today - datetime.timedelta(days=200),
        last_sale_date=today,
    )
    p_ineligible = DimProduct(
        product_id=102,
        sku="SKU-NEW-02",
        description="Brand New Pen",
        category="Stationery",
        is_product=True,
        unit_cost_est=Decimal("1.20"),
        abc_class="C",
        xyz_class="Z",
        first_sale_date=today - datetime.timedelta(days=20),  # Only 20 days span < 90
        last_sale_date=today,
    )
    p_postage = DimProduct(
        product_id=103,
        sku="POSTAGE",
        description="Postage Service Fee",
        category="Fees",
        is_product=False,
        unit_cost_est=Decimal("0.00"),
        abc_class="C",
        xyz_class="Z",
        first_sale_date=today - datetime.timedelta(days=300),
        last_sale_date=today,
    )
    ref_db_session.add_all([p_eligible, p_ineligible, p_postage])

    # 4. Customer
    cust = DimCustomer(
        customer_id=1,
        customer_key="cust-anon-1",
        country="United Kingdom",
        first_order_date=today - datetime.timedelta(days=100),
        last_order_date=today - datetime.timedelta(days=5),
    )
    ref_db_session.add(cust)

    # 5. Fact Daily Sales (60 days of sales)
    sales = []
    for d in range(60):
        dt = today - datetime.timedelta(days=d)
        sales.append(
            FactDailySales(
                date_key=dt,
                store_id=1,
                product_id=101,
                units_sold=10,
                units_returned=1,
                revenue=Decimal("100.00"),
                n_orders=5,
            )
        )
    ref_db_session.add_all(sales)

    # 6. Pipeline Run
    run_log = PipelineRun(
        run_id=uuid.uuid4(),
        flow_name="score_all",
        trigger="schedule",
        status="succeeded",
        finished_at=datetime.datetime(2026, 11, 20, 2, 0, 0, tzinfo=datetime.UTC),
    )
    ref_db_session.add(run_log)

    ref_db_session.commit()

    return {
        "viewer": viewer,
        "store": store,
        "product_eligible": p_eligible,
        "product_ineligible": p_ineligible,
        "product_postage": p_postage,
        "as_of": today,
    }


@pytest.fixture
def client(ref_db_engine: sa.Engine, seeded_db: dict[str, Any]):
    """FastAPI TestClient with overridden get_db dependency."""
    app = create_app()
    session_cls = sessionmaker(bind=ref_db_engine)

    def override_get_db():
        session = session_cls()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def auth_headers(seeded_db: dict[str, Any]) -> dict[str, str]:
    """Viewer authorization headers."""
    token = create_access_token(
        user_id=seeded_db["viewer"].user_id,
        role=seeded_db["viewer"].role,
    )
    return {"Authorization": f"Bearer {token}"}


# =========================================================================
# 1. Reference Data Endpoints Tests (API §2.2)
# =========================================================================


def test_list_stores(client: TestClient, auth_headers: dict[str, str]) -> None:
    """Verify /api/v1/stores returns list of stores."""
    resp = client.get("/api/v1/stores", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["store_code"] == "ONLINE"
    assert data[0]["store_name"] == "Online Store"


def test_list_stores_unauthenticated_returns_401(client: TestClient) -> None:
    """Verify unauthenticated request is rejected with 401 problem details."""
    resp = client.get("/api/v1/stores")
    assert resp.status_code == 401
    assert resp.headers["Content-Type"] == "application/problem+json"


def test_list_products_pagination_and_filtering(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Verify /api/v1/products supports pagination, search, and ABC filtering."""
    # 1. List all products
    resp = client.get("/api/v1/products?page=1&page_size=10", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    assert body["page"] == 1

    # 2. Search query 'ceramic'
    search_resp = client.get("/api/v1/products?q=ceramic", headers=auth_headers)
    assert search_resp.status_code == 200
    search_body = search_resp.json()
    assert search_body["total"] == 1
    assert search_body["items"][0]["sku"] == "SKU-ELIGIBLE-01"
    assert search_body["items"][0]["forecast_eligible"] is True

    # 3. Filter by abc_class 'A'
    abc_resp = client.get("/api/v1/products?abc_class=A", headers=auth_headers)
    assert abc_resp.status_code == 200
    assert abc_resp.json()["total"] == 1

    # 4. Invalid abc_class rejected with 422
    invalid_abc = client.get("/api/v1/products?abc_class=INVALID", headers=auth_headers)
    assert invalid_abc.status_code == 422

    # 5. Filter merchandise only
    merch_resp = client.get("/api/v1/products?is_product=true", headers=auth_headers)
    assert merch_resp.status_code == 200
    assert merch_resp.json()["total"] == 2


def test_get_product_detail(client: TestClient, auth_headers: dict[str, str]) -> None:
    """Verify /api/v1/products/{id} returns product profile and eligibility explanation."""
    # 1. Eligible product
    resp = client.get("/api/v1/products/101", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["product_id"] == 101
    assert data["sku"] == "SKU-ELIGIBLE-01"
    assert data["forecast_eligible"] is True

    # 2. Ineligible product explains rationale
    inelig_resp = client.get("/api/v1/products/102", headers=auth_headers)
    assert inelig_resp.status_code == 200
    inelig_data = inelig_resp.json()
    assert inelig_data["forecast_eligible"] is False
    assert "below the required 90 days" in inelig_data["eligibility_reason"]

    # 3. Unknown product returns 404
    not_found = client.get("/api/v1/products/9999", headers=auth_headers)
    assert not_found.status_code == 404
    assert not_found.json()["type"] == "urn:retailpulse:problem:not_found"


def test_get_daily_sales_validation_and_aggregation(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Verify /api/v1/sales/daily date range bounds and product/catalog aggregation."""
    # 1. Valid range for product 101
    resp = client.get(
        "/api/v1/sales/daily?from=2026-11-01&to=2026-11-20&product_id=101",
        headers=auth_headers,
    )
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 20
    assert items[0]["units_sold"] == 10

    # 2. Catalog total aggregation (omit product_id)
    cat_resp = client.get(
        "/api/v1/sales/daily?from=2026-11-01&to=2026-11-10",
        headers=auth_headers,
    )
    assert cat_resp.status_code == 200
    cat_items = cat_resp.json()
    assert len(cat_items) == 10
    assert cat_items[0]["net_units"] == 9  # 10 sold - 1 returned

    # 3. Validation: to < from returns 422
    inv_range = client.get(
        "/api/v1/sales/daily?from=2026-11-20&to=2026-11-01",
        headers=auth_headers,
    )
    assert inv_range.status_code == 422

    # 4. Validation: range > 800 days returns 422
    huge_range = client.get(
        "/api/v1/sales/daily?from=2024-01-01&to=2026-11-20",
        headers=auth_headers,
    )
    assert huge_range.status_code == 422
    assert "exceeds maximum allowed 800 days" in huge_range.json()["detail"]


# =========================================================================
# 2. Overview KPIs Endpoints Tests (API §2.3 & §3.2)
# =========================================================================


def test_get_kpis_overview_seeded_metrics(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Verify /api/v1/kpis/overview aggregates metrics and sets Cache-Control: no-store."""
    resp = client.get("/api/v1/kpis/overview", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-store"
    body = resp.json()

    assert body["currency"] == "GBP"
    assert body["data_as_of"] == "2026-11-20"
    assert Decimal(str(body["revenue_30d"])) == Decimal("3000.00")  # 30 days * 100
    assert Decimal(str(body["revenue_prev_30d"])) == Decimal("3000.00")
    assert body["revenue_change_pct"] == 0.0
    assert body["active_customers_90d"] == 1
    assert body["last_pipeline_run"] is not None
    assert body["last_pipeline_run"]["flow_name"] == "score_all"
    assert body["last_pipeline_run"]["status"] == "succeeded"


def test_get_kpis_summary_alias(client: TestClient, auth_headers: dict[str, str]) -> None:
    """Verify /api/v1/kpis/summary acts as an alias for /overview."""
    resp = client.get("/api/v1/kpis/summary", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["data_as_of"] == "2026-11-20"


# =========================================================================
# 3. Forecasts & 503 Before First Scoring Tests (API §2.3 & §3.3)
# =========================================================================


def test_forecasts_before_first_scoring_returns_503(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Verify /api/v1/forecasts returns 503 Problem Details before initial scoring."""
    # When no active output exists in ml.active_output
    resp = client.get("/api/v1/forecasts?product_id=101", headers=auth_headers)
    assert resp.status_code == 503
    assert resp.headers["Content-Type"] == "application/problem+json"
    body = resp.json()
    assert body["type"] == "urn:retailpulse:problem:model_unavailable"
    assert "No active forecast model available" in body["detail"]


def test_forecast_accuracy_before_first_scoring_returns_503(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    """Verify /api/v1/forecasts/accuracy returns 503 before initial scoring."""
    resp = client.get("/api/v1/forecasts/accuracy", headers=auth_headers)
    assert resp.status_code == 503
    assert resp.json()["type"] == "urn:retailpulse:problem:model_unavailable"


def test_forecasts_active_champion_success(
    client: TestClient,
    auth_headers: dict[str, str],
    ref_db_session: Session,
    seeded_db: dict[str, Any],
) -> None:
    """Verify /api/v1/forecasts returns full points payload once active champion is registered."""
    run_id = uuid.uuid4()
    as_of = seeded_db["as_of"]

    # 1. Register champion model run
    model_run = ModelRun(
        run_id=run_id,
        task="forecast",
        algorithm="lightgbm",
        mlflow_run_id="mlflow_run_123",
        is_champion=True,
        metrics={
            "wape": 0.35,
            "baselines": {"seasonal_naive": {"wape": 0.48}},
            "gate": {"passed": True, "reason": "wape 0.35 < 0.48"},
            "by_horizon": {"1-7": 0.30, "8-14": 0.35, "15-28": 0.40},
        },
    )
    ref_db_session.add(model_run)

    # 2. Add active_output pointer
    active_ptr = ActiveOutput(
        task="forecast",
        model_run_id=run_id,
        data_as_of=as_of,
    )
    ref_db_session.add(active_ptr)

    # 3. Add forecast rows for product 101
    f_rows = []
    for h in range(1, 4):
        target_d = as_of + datetime.timedelta(days=h)
        f_rows.append(
            Forecast(
                model_run_id=run_id,
                store_id=1,
                product_id=101,
                origin_date=as_of,
                target_date=target_d,
                horizon_day=h,
                yhat=Decimal("15.0"),
                yhat_lower=Decimal("10.0"),
                yhat_upper=Decimal("20.0"),
                interval_level=Decimal("0.80"),
            )
        )
    ref_db_session.add_all(f_rows)

    # 4. Add evaluation metrics
    metrics = [
        ForecastMetric(
            model_run_id=run_id,
            scope="global",
            scope_key="global",
            metric="wape",
            value=0.35,
        ),
        ForecastMetric(
            model_run_id=run_id,
            scope="global",
            scope_key="global",
            metric="bias",
            value=-0.02,
        ),
        ForecastMetric(
            model_run_id=run_id,
            scope="abc",
            scope_key="A",
            metric="wape",
            value=0.25,
        ),
        ForecastMetric(
            model_run_id=run_id,
            scope="horizon",
            scope_key="1-7",
            metric="wape",
            value=0.30,
        ),
    ]
    ref_db_session.add_all(metrics)
    ref_db_session.commit()
    api_cache.clear()

    # 5. Query /api/v1/forecasts for product 101 with horizon 3
    resp = client.get("/api/v1/forecasts?product_id=101&horizon=3", headers=auth_headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"]["run_id"] == str(run_id)
    assert data["model"]["algorithm"] == "lightgbm"
    assert data["product"]["product_id"] == 101
    assert data["product"]["sku"] == "SKU-ELIGIBLE-01"
    assert len(data["points"]) == 3
    assert data["points"][0]["yhat"] == 15.0
    assert data["points"][0]["yhat_lower"] == 10.0
    assert data["points"][0]["yhat_upper"] == 20.0

    # 6. Query ineligible product 102 returns 422 with explanation
    inelig_resp = client.get("/api/v1/forecasts?product_id=102", headers=auth_headers)
    assert inelig_resp.status_code == 422
    assert "not eligible for forecasting" in inelig_resp.json()["detail"]

    # 7. Query unknown product returns 404
    not_found = client.get("/api/v1/forecasts?product_id=9999", headers=auth_headers)
    assert not_found.status_code == 404

    # 8. Query /api/v1/forecasts/accuracy
    acc_resp = client.get("/api/v1/forecasts/accuracy", headers=auth_headers)
    assert acc_resp.status_code == 200
    assert acc_resp.headers["Cache-Control"] == "no-store"
    acc_data = acc_resp.json()
    assert acc_data["run_id"] == str(run_id)
    assert acc_data["algorithm"] == "lightgbm"
    assert acc_data["metrics"]["wape"] == 0.35
    assert acc_data["by_abc"]["A"]["wape"] == 0.25
    assert acc_data["gate"]["passed"] is True
    assert "0.35 < 0.48" in acc_data["gate"]["reason"]
