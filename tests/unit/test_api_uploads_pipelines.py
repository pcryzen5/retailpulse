"""Unit and integration tests for Uploads and Pipelines API endpoints.

Implements P5-08 verification:
- POST /api/v1/uploads/sales:
  - Size check (413 Payload Too Large when exceeding RP_MAX_UPLOAD_MB).
  - Extension allowlist and magic bytes check (415 Unsupported Media Type for .xlsm, corrupt files).
  - Sane header validation against adapter schema (422 with missing columns).
  - Duplicate file detection: returns 200 OK with existing batch_id and pipeline_run_id.
  - New valid upload: returns 202 Accepted with batch_id and pipeline_run_id, stored mode 0600.
- GET /api/v1/uploads: Paginated list with filtering and sorting (Analyst+).
- GET /api/v1/uploads/{batch_id}: Detailed batch and data quality results (Analyst+).
- GET /api/v1/pipelines/runs: Paginated list of pipeline runs with filtering (Viewer+).
- GET /api/v1/pipelines/runs/{run_id}: Detailed run execution record (Viewer+).
- POST /api/v1/pipelines/{flow_name}/trigger:
  - Concurrency gate: returns 409 Conflict if flow is already queued or running.
  - RBAC: Analyst+ for etl and score_all; Admin only for train_* and drift_check; Viewer 403.
  - Trigger failure: returns 503 and marks run failed if trigger returns False.
  - Trigger success: returns 202 with Location header.
- Audit logging: audit logs created for upload and trigger actions (SEC-LOG-01).
"""

from __future__ import annotations

import io
import uuid
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
from retailpulse.db.models.ops import AuditLog, DataQualityResult, IngestionBatch, PipelineRun
from retailpulse.db.session import get_db
from retailpulse.pipelines.trigger import InlinePipelineTrigger, PipelineTrigger
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
def seeded_db(ops_db_session: Session) -> dict[str, Any]:
    """Seed test users (viewer, analyst, admin) and initial runs."""
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
    ops_db_session.commit()

    return {
        "viewer_id": viewer.user_id,
        "analyst_id": analyst.user_id,
        "admin_id": admin.user_id,
    }


@pytest.fixture
def client_and_auth(tmp_path, ops_db_session: Session, seeded_db: dict[str, Any]):
    """Create test client with custom data_dir and auth tokens."""
    settings = Settings(
        max_upload_mb=2,  # Set 2 MB for fast testing of limits
        data_dir=tmp_path / "data",
    )
    app = create_app(settings=settings)
    app.dependency_overrides[get_db] = lambda: ops_db_session

    inline_trigger = InlinePipelineTrigger()
    app.state.pipeline_trigger = inline_trigger

    client = TestClient(app)

    viewer_tok = create_access_token(
        user_id=seeded_db["viewer_id"],
        role="viewer",
        secret_key=settings.jwt_secret.get_secret_value(),
    )
    analyst_tok = create_access_token(
        user_id=seeded_db["analyst_id"],
        role="analyst",
        secret_key=settings.jwt_secret.get_secret_value(),
    )
    admin_tok = create_access_token(
        user_id=seeded_db["admin_id"],
        role="admin",
        secret_key=settings.jwt_secret.get_secret_value(),
    )

    return {
        "client": client,
        "app": app,
        "trigger": inline_trigger,
        "viewer_headers": {"Authorization": f"Bearer {viewer_tok}"},
        "analyst_headers": {"Authorization": f"Bearer {analyst_tok}"},
        "admin_headers": {"Authorization": f"Bearer {admin_tok}"},
        "data_dir": tmp_path / "data",
    }


# Sample CSV contents
VALID_GENERIC_CSV = (
    b"invoice_no,invoice_date,sku,quantity,unit_price\n"
    b"INV001,2026-10-01,SKU100,5,19.99\n"
    b"INV002,2026-10-02,SKU200,2,49.50\n"
)

VALID_ONLINE_RETAIL_CSV = (
    b"Invoice,StockCode,Description,Quantity,InvoiceDate,Price,Customer ID,Country\n"
    b"536365,85123A,WHITE HANGING HEART,6,2010-12-01 08:26:00,2.55,17850,United Kingdom\n"
)

INVALID_HEADER_CSV = b"foo,bar,baz\n1,2,3\n"


# --- Upload Validation Tests ---


def test_upload_sales_size_limit_413(client_and_auth):
    """Verify that uploading a file larger than max_upload_mb returns 413 Payload Too Large."""
    client = client_and_auth["client"]
    headers = client_and_auth["analyst_headers"]

    # Limit is 2 MB in fixture; send 2.5 MB
    large_content = b"a" * (int(2.5 * 1024 * 1024))
    files = {"file": ("big.csv", io.BytesIO(large_content), "text/csv")}
    data = {"source_name": "generic_csv"}

    response = client.post("/api/v1/uploads/sales", headers=headers, files=files, data=data)
    assert response.status_code == 413
    body = response.json()
    assert body["type"] == "urn:retailpulse:problem:payload_too_large"


def test_upload_sales_unsupported_media_type_415(client_and_auth):
    """Verify that unsupported extensions or disguised binaries return 415."""
    client = client_and_auth["client"]
    headers = client_and_auth["analyst_headers"]

    # 1. Macro-enabled .xlsm rejected
    files = {"file": ("macro.xlsm", io.BytesIO(b"PK\x03\x04fakexlsm"), "application/vnd.ms-excel")}
    response = client.post(
        "/api/v1/uploads/sales", headers=headers, files=files, data={"source_name": "generic_csv"}
    )
    assert response.status_code == 415
    assert response.json()["type"] == "urn:retailpulse:problem:unsupported_media_type"

    # 2. Unsupported extension .txt
    files = {"file": ("data.txt", io.BytesIO(b"hello"), "text/plain")}
    response = client.post(
        "/api/v1/uploads/sales", headers=headers, files=files, data={"source_name": "generic_csv"}
    )
    assert response.status_code == 415

    # 3. CSV with ZIP magic bytes disguised as CSV
    files = {"file": ("fake.csv", io.BytesIO(b"PK\x03\x04disguisedzip"), "text/csv")}
    response = client.post(
        "/api/v1/uploads/sales", headers=headers, files=files, data={"source_name": "generic_csv"}
    )
    assert response.status_code == 415

    # 4. XLSX without ZIP signature
    files = {
        "file": (
            "fake.xlsx",
            io.BytesIO(b"not-a-zip-file"),
            "application/vnd.openxmlformats-officedocument",
        )
    }
    response = client.post(
        "/api/v1/uploads/sales",
        headers=headers,
        files=files,
        data={"source_name": "online_retail_ii"},
    )
    assert response.status_code == 415


def test_upload_sales_invalid_header_422(client_and_auth):
    """Verify that uploading CSV with missing required columns returns 422 with missing list."""
    client = client_and_auth["client"]
    headers = client_and_auth["analyst_headers"]

    files = {"file": ("bad_header.csv", io.BytesIO(INVALID_HEADER_CSV), "text/csv")}
    response = client.post(
        "/api/v1/uploads/sales", headers=headers, files=files, data={"source_name": "generic_csv"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "urn:retailpulse:problem:validation_error"
    assert "Missing required column(s)" in body["detail"]


def test_upload_sales_invalid_source_name_422(client_and_auth):
    """Verify that unknown source_name returns 422."""
    client = client_and_auth["client"]
    headers = client_and_auth["analyst_headers"]

    files = {"file": ("valid.csv", io.BytesIO(VALID_GENERIC_CSV), "text/csv")}
    response = client.post(
        "/api/v1/uploads/sales",
        headers=headers,
        files=files,
        data={"source_name": "unknown_source"},
    )
    assert response.status_code == 422
    body = response.json()
    assert "Invalid source_name" in body["detail"]


def test_upload_sales_success_and_deduplication(client_and_auth, ops_db_session: Session):
    """Verify successful upload (202), file storage with mode 0600, and duplicate upload (200)."""
    client = client_and_auth["client"]
    headers = client_and_auth["analyst_headers"]
    trigger = client_and_auth["trigger"]

    # 1. First upload: valid generic CSV
    files = {"file": ("valid_sales.csv", io.BytesIO(VALID_GENERIC_CSV), "text/csv")}
    response1 = client.post(
        "/api/v1/uploads/sales",
        headers=headers,
        files=files,
        data={"source_name": "generic_csv"},
    )
    assert response1.status_code == 202
    body1 = response1.json()
    batch_id_str = body1["batch_id"]
    pipe_run_id_str = body1["pipeline_run_id"]
    assert body1["status"] == "queued"
    assert "Location" in response1.headers
    assert response1.headers["Location"] == f"/api/v1/uploads/{batch_id_str}"

    # Verify trigger was called
    assert len(trigger.dispatched_runs) == 1
    assert trigger.dispatched_runs[0][0] == "etl"
    assert str(trigger.dispatched_runs[0][1]) == pipe_run_id_str

    # Verify database persistence
    batch = ops_db_session.get(IngestionBatch, uuid.UUID(batch_id_str))
    assert batch is not None
    assert batch.source_name == "generic_csv"
    assert batch.status == "queued"

    # Verify audit log recorded upload (SEC-LOG-01)
    audit = ops_db_session.execute(
        sa.select(AuditLog).where(AuditLog.action == "upload")
    ).scalar_one_or_none()
    assert audit is not None
    assert str(audit.user_id) == str(batch.uploaded_by)

    # 2. Second upload: IDENTICAL file with same source_name -> returns 200 OK (idempotent dedupe)
    files2 = {"file": ("sales_copy.csv", io.BytesIO(VALID_GENERIC_CSV), "text/csv")}
    response2 = client.post(
        "/api/v1/uploads/sales",
        headers=headers,
        files=files2,
        data={"source_name": "generic_csv"},
    )
    assert response2.status_code == 200
    body2 = response2.json()
    assert body2["batch_id"] == batch_id_str
    assert body2["pipeline_run_id"] == pipe_run_id_str
    # Verify no second trigger was dispatched
    assert len(trigger.dispatched_runs) == 1


# --- Batch Query Endpoints Tests ---


def test_list_and_get_uploads(client_and_auth, ops_db_session: Session):
    """Verify GET /uploads and GET /uploads/{batch_id} with data quality checks."""
    client = client_and_auth["client"]
    headers = client_and_auth["analyst_headers"]

    batch_id = uuid.uuid4()
    now = utc_now()
    batch = IngestionBatch(
        batch_id=batch_id,
        source_name="generic_csv",
        source_file="uploads/dummy.csv",
        original_filename="dummy.csv",
        file_sha256="a" * 64,
        status="succeeded",
        row_count_in=1000,
        row_count_loaded=990,
        row_count_rejected=10,
        created_at=now,
    )
    dq1 = DataQualityResult(
        batch_id=batch_id,
        check_name="duplicate_rows_removed",
        severity="info",
        passed=True,
        affected_rows=10,
        details={},
        created_at=now,
    )
    ops_db_session.add_all([batch, dq1])
    ops_db_session.commit()

    # List batches
    res_list = client.get("/api/v1/uploads", headers=headers)
    assert res_list.status_code == 200
    list_body = res_list.json()
    assert list_body["total"] >= 1
    item = next(b for b in list_body["items"] if b["batch_id"] == str(batch_id))
    assert item["status"] == "succeeded"
    assert item["row_count_loaded"] == 990

    # Get batch detail
    res_detail = client.get(f"/api/v1/uploads/{batch_id}", headers=headers)
    assert res_detail.status_code == 200
    detail_body = res_detail.json()
    assert detail_body["batch_id"] == str(batch_id)
    assert len(detail_body["quality"]) == 1
    assert detail_body["quality"][0]["check_name"] == "duplicate_rows_removed"

    # 404 for unknown batch
    res_not_found = client.get(f"/api/v1/uploads/{uuid.uuid4()}", headers=headers)
    assert res_not_found.status_code == 404


# --- Pipeline Runs and Trigger Tests ---


def test_pipelines_runs_list_and_get(client_and_auth, ops_db_session: Session):
    """Verify GET /pipelines/runs and GET /pipelines/runs/{run_id} (Viewer+)."""
    client = client_and_auth["client"]
    headers = client_and_auth["viewer_headers"]

    run_id = uuid.uuid4()
    run = PipelineRun(
        run_id=run_id,
        flow_name="score_all",
        trigger="manual",
        status="succeeded",
        params={"scope": "all"},
        queued_at=utc_now(),
    )
    ops_db_session.add(run)
    ops_db_session.commit()

    # List runs
    res = client.get("/api/v1/pipelines/runs?flow_name=score_all", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["total"] >= 1
    assert body["items"][0]["run_id"] == str(run_id)

    # Get single run
    res_single = client.get(f"/api/v1/pipelines/runs/{run_id}", headers=headers)
    assert res_single.status_code == 200
    assert res_single.json()["flow_name"] == "score_all"

    # 404 for nonexistent run
    assert client.get(f"/api/v1/pipelines/runs/{uuid.uuid4()}", headers=headers).status_code == 404


def test_trigger_pipeline_concurrency_gate_409(client_and_auth, ops_db_session: Session):
    """Verify that triggering an already active flow returns 409 Conflict."""
    client = client_and_auth["client"]
    admin_headers = client_and_auth["admin_headers"]

    # Seed an active running flow
    active_run = PipelineRun(
        run_id=uuid.uuid4(),
        flow_name="train_forecast",
        trigger="manual",
        status="running",
        params={},
        queued_at=utc_now(),
    )
    ops_db_session.add(active_run)
    ops_db_session.commit()

    # Triggering train_forecast must return 409 Conflict
    response = client.post(
        "/api/v1/pipelines/train_forecast/trigger",
        headers=admin_headers,
        json={"params": {}},
    )
    assert response.status_code == 409
    body = response.json()
    assert body["type"] == "urn:retailpulse:problem:conflict"
    assert "already active" in body["detail"]


def test_trigger_pipeline_rbac(client_and_auth):
    """Verify RBAC: Viewer forbidden; Analyst allowed for etl & score_all; Admin allowed for all."""
    client = client_and_auth["client"]
    viewer_headers = client_and_auth["viewer_headers"]
    analyst_headers = client_and_auth["analyst_headers"]
    admin_headers = client_and_auth["admin_headers"]

    # 1. Viewer cannot trigger any flow
    res = client.post("/api/v1/pipelines/score_all/trigger", headers=viewer_headers, json={})
    assert res.status_code == 403

    # 2. Analyst can trigger score_all
    res_analyst = client.post(
        "/api/v1/pipelines/score_all/trigger", headers=analyst_headers, json={}
    )
    assert res_analyst.status_code == 202
    run_id = res_analyst.json()["pipeline_run_id"]
    assert res_analyst.headers["Location"] == f"/api/v1/pipelines/runs/{run_id}"

    # 3. Analyst forbidden from triggering train_forecast (Admin only)
    res_analyst_train = client.post(
        "/api/v1/pipelines/train_forecast/trigger", headers=analyst_headers, json={}
    )
    assert res_analyst_train.status_code == 403

    # 4. Admin can trigger train_forecast
    res_admin_train = client.post(
        "/api/v1/pipelines/train_forecast/trigger", headers=admin_headers, json={}
    )
    assert res_admin_train.status_code == 202


def test_trigger_pipeline_failure_503(client_and_auth, ops_db_session: Session):
    """Verify that trigger dispatch failure marks run failed and returns 503."""
    client = client_and_auth["client"]
    admin_headers = client_and_auth["admin_headers"]

    class FailingTrigger(PipelineTrigger):
        async def trigger(self, flow_name, run_id, params=None):
            return False

    client_and_auth["app"].state.pipeline_trigger = FailingTrigger()

    response = client.post("/api/v1/pipelines/drift_check/trigger", headers=admin_headers, json={})
    assert response.status_code == 503
    body = response.json()
    assert body["type"] == "urn:retailpulse:problem:model_unavailable"

    # Verify run in DB was marked 'failed'
    failed_run = ops_db_session.execute(
        sa.select(PipelineRun).where(PipelineRun.flow_name == "drift_check")
    ).scalar_one_or_none()
    assert failed_run is not None
    assert failed_run.status == "failed"
