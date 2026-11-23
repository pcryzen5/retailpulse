"""Unit and integration tests for Models, Monitoring, and Alerts API endpoints.

Implements P5-07 verification:
- GET /api/v1/models: Filtered and paginated list; mlflow_url hidden from non-admins (Viewer+).
- GET /api/v1/models/{run_id}: Run details; mlflow_url shown to Admins only (Viewer+).
- POST /api/v1/models/{task}/train: Trigger training pipeline;
  202 with Location (Admin only, 403 for Viewer/Analyst).
- POST /api/v1/models/{run_id}/promote: Move champion alias, trigger score_all,
  enforce gate (Admin only, 403 for Viewer/Analyst).
- GET /api/v1/monitoring/drift: List drift reports (Viewer+).
- GET /api/v1/monitoring/drift/{report_id}: Report summary JSON (Viewer+).
- GET /api/v1/monitoring/drift/{report_id}/html: Evidently HTML with sandbox CSP headers (Viewer+).
- GET /api/v1/monitoring/performance: Realized performance snapshots (Viewer+).
- GET /api/v1/alerts: Filtered alerts by status and severity (Viewer+).
- POST /api/v1/alerts/{alert_id}/ack: Acknowledge alert, audit-logged
  (Analyst+ only, 403 for Viewer).
- Audit log verification: trigger, promote, alert_ack in ops.audit_log (SEC-LOG-01).
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
from retailpulse.db.models.ml import ModelRun
from retailpulse.db.models.ops import Alert, AuditLog, DriftReport, ModelPerformance, PipelineRun
from retailpulse.db.session import get_db
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password
from retailpulse.timeutil import utc_now


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def ops_db_engine():
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
def ops_db_session(ops_db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=ops_db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def seeded_ops_db(ops_db_session: Session) -> dict[str, Any]:
    """Seed test users (viewer, analyst, admin), model runs, drift reports, performance, alerts."""
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
    ops_db_session.add_all([viewer, analyst, admin])

    # 2. Model runs
    run_champ = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="lightgbm",
        mlflow_run_id="mlflow_champ_1",
        status="succeeded",
        is_champion=True,
        trained_at=now - datetime.timedelta(days=7),
        data_end=today - datetime.timedelta(days=7),
        metrics={"wape": 0.42, "gate_passed": True, "gate_reason": "Beats baseline"},
        params={"objective": "tweedie", "horizon": 28},
        git_sha="gitsha001",
        dataset_hash="hash001",
        seed=42,
    )
    run_candidate = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="tft",
        mlflow_run_id="mlflow_cand_2",
        status="succeeded",
        is_champion=False,
        trained_at=now - datetime.timedelta(days=1),
        data_end=today - datetime.timedelta(days=1),
        metrics={"wape": 0.38, "gate_passed": True, "gate_reason": "WAPE 0.38 < 0.42"},
        params={"learning_rate": 0.01, "horizon": 28},
        git_sha="gitsha002",
        dataset_hash="hash002",
        seed=42,
    )
    run_failed_gate = ModelRun(
        run_id=uuid.uuid4(),
        task="forecast",
        algorithm="linear",
        mlflow_run_id="mlflow_fail_3",
        status="succeeded",
        is_champion=False,
        trained_at=now - datetime.timedelta(days=2),
        data_end=today - datetime.timedelta(days=2),
        metrics={"wape": 0.75, "gate_passed": False, "gate_reason": "WAPE 0.75 exceeds threshold"},
        params={"alpha": 1.0},
        git_sha="gitsha003",
        dataset_hash="hash003",
        seed=42,
    )
    run_errored = ModelRun(
        run_id=uuid.uuid4(),
        task="churn",
        algorithm="lightgbm",
        mlflow_run_id="mlflow_err_4",
        status="failed",
        is_champion=False,
        trained_at=now - datetime.timedelta(days=3),
        data_end=today - datetime.timedelta(days=3),
        metrics={},
        params={},
        git_sha="gitsha004",
        dataset_hash="hash004",
        seed=42,
    )
    ops_db_session.add_all([run_champ, run_candidate, run_failed_gate, run_errored])

    # 3. Drift reports
    report_data = DriftReport(
        report_id=uuid.uuid4(),
        task="forecast",
        model_run_id=run_champ.run_id,
        kind="data",
        reference_start=today - datetime.timedelta(days=60),
        reference_end=today - datetime.timedelta(days=30),
        current_start=today - datetime.timedelta(days=29),
        current_end=today,
        dataset_drift=True,
        share_drifted_features=Decimal("0.2500"),
        summary={
            "n_features": 20,
            "n_drifted": 5,
            "html_content": "<html><body><h1>Data Drift Report</h1></body></html>",
        },
        report_path="drift/forecast_data_20261120.html",
        created_at=now - datetime.timedelta(days=1),
    )
    report_pred = DriftReport(
        report_id=uuid.uuid4(),
        task="churn",
        model_run_id=None,
        kind="prediction",
        reference_start=today - datetime.timedelta(days=60),
        reference_end=today - datetime.timedelta(days=30),
        current_start=today - datetime.timedelta(days=29),
        current_end=today,
        dataset_drift=False,
        share_drifted_features=Decimal("0.0000"),
        summary={"ks_stat": 0.04, "p_value": 0.32},
        report_path="drift/churn_pred_20261120.html",
        created_at=now,
    )
    ops_db_session.add_all([report_data, report_pred])

    # 4. Model Performance snapshots
    perf1 = ModelPerformance(
        model_run_id=run_champ.run_id,
        task="forecast",
        window_start=today - datetime.timedelta(days=14),
        window_end=today,
        metric="realized_wape",
        value=0.44,
        baseline_value=0.51,
        n_observations=1400,
        created_at=now,
    )
    ops_db_session.add(perf1)

    # 5. Alerts
    alert_open = Alert(
        severity="warning",
        kind="forecast_degraded",
        message="Realized WAPE 0.44 exceeded 1.25x backtest WAPE 0.42",
        payload={"realized_wape": 0.44, "backtest_wape": 0.42},
        dedupe_key="forecast_degraded_20261120",
        created_at=now - datetime.timedelta(hours=2),
        acknowledged_at=None,
    )
    alert_ack = Alert(
        severity="info",
        kind="drift_detected",
        message="5 features exhibited statistically significant distribution drift",
        payload={"share_drifted": 0.25},
        dedupe_key="drift_forecast_20261119",
        created_at=now - datetime.timedelta(days=1),
        acknowledged_at=now - datetime.timedelta(hours=10),
        acknowledged_by=analyst.user_id,
    )
    ops_db_session.add_all([alert_open, alert_ack])

    ops_db_session.commit()

    return {
        "viewer_id": viewer.user_id,
        "analyst_id": analyst.user_id,
        "admin_id": admin.user_id,
        "champion_id": run_champ.run_id,
        "candidate_id": run_candidate.run_id,
        "failed_gate_id": run_failed_gate.run_id,
        "errored_id": run_errored.run_id,
        "report_data_id": report_data.report_id,
        "report_pred_id": report_pred.report_id,
        "alert_open_id": alert_open.alert_id,
        "alert_ack_id": alert_ack.alert_id,
    }


@pytest.fixture
def client_and_auth(ops_db_session: Session, seeded_ops_db: dict[str, Any]):
    """Create test client and Bearer auth headers for viewer, analyst, and admin."""
    app = create_app()
    app.dependency_overrides[get_db] = lambda: ops_db_session

    client = TestClient(app)

    viewer_token = create_access_token(
        user_id=seeded_ops_db["viewer_id"],
        role="viewer",
    )
    analyst_token = create_access_token(
        user_id=seeded_ops_db["analyst_id"],
        role="analyst",
    )
    admin_token = create_access_token(
        user_id=seeded_ops_db["admin_id"],
        role="admin",
    )

    return {
        "client": client,
        "viewer": {"Authorization": f"Bearer {viewer_token}"},
        "analyst": {"Authorization": f"Bearer {analyst_token}"},
        "admin": {"Authorization": f"Bearer {admin_token}"},
    }


# --- Model Tests ---


def test_list_models_rbac_and_filters(client_and_auth, seeded_ops_db):
    """Test GET /api/v1/models listing, filtering, and mlflow_url RBAC redaction."""
    client = client_and_auth["client"]

    # 1. Viewer sees all 4 models but mlflow_url is None
    resp_viewer = client.get("/api/v1/models", headers=client_and_auth["viewer"])
    assert resp_viewer.status_code == 200
    body = resp_viewer.json()
    assert body["total"] == 4
    for item in body["items"]:
        assert item["mlflow_url"] is None

    # 2. Admin sees mlflow_url populated
    resp_admin = client.get("/api/v1/models", headers=client_and_auth["admin"])
    assert resp_admin.status_code == 200
    items = resp_admin.json()["items"]
    assert any(item["mlflow_url"] is not None for item in items)

    # 3. Filter by task and is_champion
    resp_filter = client.get(
        "/api/v1/models?task=forecast&is_champion=true",
        headers=client_and_auth["viewer"],
    )
    assert resp_filter.status_code == 200
    filtered_items = resp_filter.json()["items"]
    assert len(filtered_items) == 1
    assert filtered_items[0]["run_id"] == str(seeded_ops_db["champion_id"])
    assert filtered_items[0]["is_champion"] is True


def test_get_model_run_detail_rbac(client_and_auth, seeded_ops_db):
    """Test GET /api/v1/models/{run_id} details and mlflow_url authorization."""
    client = client_and_auth["client"]
    champ_id = seeded_ops_db["champion_id"]

    # 1. 404 for non-existent model
    resp_404 = client.get(
        f"/api/v1/models/{uuid.uuid4()}",
        headers=client_and_auth["viewer"],
    )
    assert resp_404.status_code == 404

    # 2. Viewer: 200 OK with mlflow_url=None
    resp_viewer = client.get(
        f"/api/v1/models/{champ_id}",
        headers=client_and_auth["viewer"],
    )
    assert resp_viewer.status_code == 200
    body_viewer = resp_viewer.json()
    assert body_viewer["run_id"] == str(champ_id)
    assert body_viewer["is_champion"] is True
    assert body_viewer["mlflow_url"] is None

    # 3. Admin: 200 OK with mlflow_url present
    resp_admin = client.get(
        f"/api/v1/models/{champ_id}",
        headers=client_and_auth["admin"],
    )
    assert resp_admin.status_code == 200
    body_admin = resp_admin.json()
    assert body_admin["mlflow_url"] is not None
    assert "mlflow_champ_1" in body_admin["mlflow_url"]


def test_train_model_rbac_and_dispatch(client_and_auth, ops_db_session: Session):
    """Test POST /api/v1/models/{task}/train RBAC, 202 Location, and audit logging."""
    client = client_and_auth["client"]

    # 1. Viewer forbidden (403)
    resp_viewer = client.post(
        "/api/v1/models/forecast/train",
        json={"algorithms": ["lightgbm"]},
        headers=client_and_auth["viewer"],
    )
    assert resp_viewer.status_code == 403

    # 2. Analyst forbidden (403)
    resp_analyst = client.post(
        "/api/v1/models/forecast/train",
        json={"algorithms": ["lightgbm"]},
        headers=client_and_auth["analyst"],
    )
    assert resp_analyst.status_code == 403

    # 3. Invalid task -> 422
    resp_inv = client.post(
        "/api/v1/models/invalid_task/train",
        json={},
        headers=client_and_auth["admin"],
    )
    assert resp_inv.status_code == 422

    # 4. Admin allowed -> 202 Accepted with Location header
    resp_admin = client.post(
        "/api/v1/models/forecast/train",
        json={"algorithms": ["lightgbm"], "params": {"n_estimators": 100}},
        headers=client_and_auth["admin"],
    )
    assert resp_admin.status_code == 202
    body = resp_admin.json()
    assert body["flow_name"] == "train_forecast"
    assert body["status"] == "queued"
    pipe_id = body["pipeline_run_id"]
    assert resp_admin.headers["Location"] == f"/api/v1/pipelines/runs/{pipe_id}"

    # Verify PipelineRun in DB
    pipe_run = ops_db_session.get(PipelineRun, uuid.UUID(pipe_id))
    assert pipe_run is not None
    assert pipe_run.flow_name == "train_forecast"
    assert pipe_run.status == "queued"

    # Verify AuditLog in DB (SEC-LOG-01)
    audit = ops_db_session.execute(
        sa.select(AuditLog).where(
            AuditLog.action == "trigger",
            AuditLog.resource == "pipeline:train_forecast",
        )
    ).scalar_one_or_none()
    assert audit is not None
    assert audit.details["task"] == "forecast"

    # 5. Duplicate while active returns 409 Conflict
    resp_dup = client.post(
        "/api/v1/models/forecast/train",
        json={},
        headers=client_and_auth["admin"],
    )
    assert resp_dup.status_code == 409


def test_promote_model_rbac_and_gates(client_and_auth, seeded_ops_db, ops_db_session: Session):
    """Test POST /api/v1/models/{run_id}/promote gates, RBAC, and score_all trigger."""
    client = client_and_auth["client"]
    cand_id = seeded_ops_db["candidate_id"]
    failed_gate_id = seeded_ops_db["failed_gate_id"]
    errored_id = seeded_ops_db["errored_id"]
    champ_id = seeded_ops_db["champion_id"]

    # 1. Viewer and Analyst forbidden (403)
    assert (
        client.post(
            f"/api/v1/models/{cand_id}/promote",
            headers=client_and_auth["viewer"],
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/v1/models/{cand_id}/promote",
            headers=client_and_auth["analyst"],
        ).status_code
        == 403
    )

    # 2. Non-succeeded model returns 409 Conflict
    resp_err = client.post(
        f"/api/v1/models/{errored_id}/promote",
        headers=client_and_auth["admin"],
    )
    assert resp_err.status_code == 409
    assert "status 'failed'" in resp_err.json()["detail"]

    # 3. Model that failed evaluation gate returns 409 Conflict
    resp_gate = client.post(
        f"/api/v1/models/{failed_gate_id}/promote",
        headers=client_and_auth["admin"],
    )
    assert resp_gate.status_code == 409
    assert "did not pass the promotion gate" in resp_gate.json()["detail"]

    # 4. Admin promotes successful candidate
    resp_promote = client.post(
        f"/api/v1/models/{cand_id}/promote",
        headers=client_and_auth["admin"],
    )
    assert resp_promote.status_code == 200
    promote_body = resp_promote.json()
    assert promote_body["task"] == "forecast"
    assert promote_body["run_id"] == str(cand_id)
    assert promote_body["previous_champion_run_id"] == str(champ_id)
    assert "score_all_pipeline_run_id" in promote_body

    # Verify DB state
    old_champ = ops_db_session.get(ModelRun, champ_id)
    new_champ = ops_db_session.get(ModelRun, cand_id)
    assert old_champ is not None
    assert old_champ.is_champion is False
    assert new_champ is not None
    assert new_champ.is_champion is True

    # Verify score_all pipeline queued
    score_id = uuid.UUID(promote_body["score_all_pipeline_run_id"])
    score_run = ops_db_session.get(PipelineRun, score_id)
    assert score_run is not None
    assert score_run.flow_name == "score_all"
    assert score_run.trigger == "promote"

    # Verify AuditLog (SEC-LOG-01)
    audit = ops_db_session.execute(
        sa.select(AuditLog).where(
            AuditLog.action == "promote",
            AuditLog.resource == f"model:forecast:{cand_id}",
        )
    ).scalar_one_or_none()
    assert audit is not None
    assert audit.details["previous_champion_run_id"] == str(champ_id)


# --- Monitoring Tests ---


def test_monitoring_drift_endpoints(client_and_auth, seeded_ops_db):
    """Test GET /api/v1/monitoring/drift, detail JSON, and HTML with CSP headers."""
    client = client_and_auth["client"]
    report_id = seeded_ops_db["report_data_id"]

    # 1. List drift reports
    resp = client.get("/api/v1/monitoring/drift", headers=client_and_auth["viewer"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert len(body["items"]) == 2

    # 2. Filter by kind
    resp_kind = client.get(
        "/api/v1/monitoring/drift?kind=prediction",
        headers=client_and_auth["viewer"],
    )
    assert resp_kind.status_code == 200
    assert resp_kind.json()["total"] == 1
    assert resp_kind.json()["items"][0]["kind"] == "prediction"

    # 3. Report detail
    resp_detail = client.get(
        f"/api/v1/monitoring/drift/{report_id}",
        headers=client_and_auth["viewer"],
    )
    assert resp_detail.status_code == 200
    detail = resp_detail.json()
    assert detail["report_id"] == str(report_id)
    assert detail["dataset_drift"] is True
    assert "n_features" in detail["summary"]

    # 4. Report HTML with CSP headers (API §2.6)
    resp_html = client.get(
        f"/api/v1/monitoring/drift/{report_id}/html",
        headers=client_and_auth["viewer"],
    )
    assert resp_html.status_code == 200
    assert "text/html" in resp_html.headers["content-type"]
    assert resp_html.headers["content-security-policy"] == "sandbox"
    assert resp_html.headers["x-content-type-options"] == "nosniff"
    assert "Data Drift Report" in resp_html.text


def test_monitoring_performance_snapshots(client_and_auth):
    """Test GET /api/v1/monitoring/performance listing and filters."""
    client = client_and_auth["client"]

    resp = client.get("/api/v1/monitoring/performance", headers=client_and_auth["viewer"])
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    item = body["items"][0]
    assert item["task"] == "forecast"
    assert item["metric"] == "realized_wape"
    assert item["value"] == 0.44
    assert item["baseline_value"] == 0.51


# --- Alert Tests ---


def test_alerts_list_and_ack(client_and_auth, seeded_ops_db, ops_db_session: Session):
    """Test GET /api/v1/alerts and POST /api/v1/alerts/{alert_id}/ack RBAC and audit."""
    client = client_and_auth["client"]
    open_id = seeded_ops_db["alert_open_id"]
    ack_id = seeded_ops_db["alert_ack_id"]

    # 1. Default list returns open alerts only
    resp_open = client.get("/api/v1/alerts", headers=client_and_auth["viewer"])
    assert resp_open.status_code == 200
    assert resp_open.json()["total"] == 1
    assert resp_open.json()["items"][0]["alert_id"] == open_id

    # 2. Filter status=all
    resp_all = client.get("/api/v1/alerts?status=all", headers=client_and_auth["viewer"])
    assert resp_all.status_code == 200
    assert resp_all.json()["total"] == 2

    # 3. Filter status=acknowledged
    resp_ack = client.get("/api/v1/alerts?status=acknowledged", headers=client_and_auth["viewer"])
    assert resp_ack.status_code == 200
    assert resp_ack.json()["total"] == 1
    assert resp_ack.json()["items"][0]["alert_id"] == ack_id

    # 4. Viewer cannot acknowledge alert (403)
    resp_v_ack = client.post(
        f"/api/v1/alerts/{open_id}/ack",
        headers=client_and_auth["viewer"],
    )
    assert resp_v_ack.status_code == 403

    # 5. Analyst acknowledges open alert (200)
    resp_a_ack = client.post(
        f"/api/v1/alerts/{open_id}/ack",
        headers=client_and_auth["analyst"],
    )
    assert resp_a_ack.status_code == 200
    body = resp_a_ack.json()
    assert body["alert_id"] == open_id
    assert body["acknowledged_at"] is not None
    assert body["acknowledged_by"] == str(seeded_ops_db["analyst_id"])

    # Verify DB state
    alert = ops_db_session.get(Alert, open_id)
    assert alert is not None
    assert alert.acknowledged_at is not None
    assert alert.acknowledged_by == seeded_ops_db["analyst_id"]

    # Verify AuditLog (SEC-LOG-01)
    audit = ops_db_session.execute(
        sa.select(AuditLog).where(
            AuditLog.action == "alert_ack",
            AuditLog.resource == f"alert:{open_id}",
        )
    ).scalar_one_or_none()
    assert audit is not None
    assert audit.user_id == seeded_ops_db["analyst_id"]
    assert audit.details["kind"] == "forecast_degraded"
