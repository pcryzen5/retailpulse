"""Unit tests for model promotion and rollback orchestration service and API endpoints.

Implements P7-03 verification per ARCHITECTURE §7, TRD §6, and API §3.10:
- Service moves alias in MLflow registry (champion -> previous_champion, new version -> champion).
- Atomically flips is_champion in ml.model_run.
- Audit-logs the promotion/rollback action (SEC-LOG-01).
- Triggers score_all pipeline run to refresh predictions.
- Refuses candidate runs that failed the promotion gate (ModelGateError -> HTTP 409 Conflict).
- Refuses non-succeeded runs (ConflictError -> HTTP 409 Conflict).
- Rollback restores previous champion (via alias, audit log, or fallback),
  audit-logs with is_rollback=True, and triggers score_all.
- Refuses rollback when no previous champion exists (ConflictError -> HTTP 409 Conflict).
- Admin-only RBAC enforcement (403 Forbidden for Viewer/Analyst).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

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
from retailpulse.db.models.ops import AuditLog, PipelineRun
from retailpulse.db.session import get_db
from retailpulse.errors import ConflictError, DomainValidationError, ModelGateError, NotFoundError
from retailpulse.pipelines.promotion import (
    PromoteResult,
    promote_model_run,
    rollback_model_run,
)
from retailpulse.pipelines.trigger import PipelineTrigger
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password
from retailpulse.timeutil import utc_now


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def db_engine():
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
def db_session(db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_pipeline_trigger() -> PipelineTrigger:
    """Mock PipelineTrigger instance."""
    trigger = MagicMock(spec=PipelineTrigger)
    trigger.trigger = AsyncMock(return_value={"status": "dispatched", "run_id": str(uuid.uuid4())})
    return trigger


@pytest.fixture
def seeded_models(db_session: Session) -> dict[str, Any]:
    """Seed champion, successful candidate, failed gate candidate, and failed status runs."""
    now = utc_now()
    admin_id = uuid.uuid4()
    admin = AppUser(
        user_id=admin_id,
        email="admin@retailpulse.local",
        password_hash=hash_password("AdminPassword123!"),
        role="admin",
        is_active=True,
    )
    viewer_id = uuid.uuid4()
    viewer = AppUser(
        user_id=viewer_id,
        email="viewer@retailpulse.local",
        password_hash=hash_password("ViewerPassword123!"),
        role="viewer",
        is_active=True,
    )
    analyst_id = uuid.uuid4()
    analyst = AppUser(
        user_id=analyst_id,
        email="analyst@retailpulse.local",
        password_hash=hash_password("AnalystPassword123!"),
        role="analyst",
        is_active=True,
    )
    db_session.add_all([admin, viewer, analyst])

    # 1. Existing champion (version 1)
    champ_id = uuid.uuid4()
    champ = ModelRun(
        run_id=champ_id,
        task="forecast",
        algorithm="seasonal_naive",
        status="succeeded",
        is_champion=True,
        trained_at=now - datetime.timedelta(days=7),
        metrics={"wape": 0.20, "gate_passed": True},
        params={"window": 14},
        git_sha="abcdef1",
        dataset_hash="hash1",
        seed=42,
        mlflow_run_id="mlflow-champ-1",
        mlflow_model_name="retailpulse-forecast",
        mlflow_model_version="1",
    )

    # 2. Candidate that passed gate (version 2)
    candidate_id = uuid.uuid4()
    candidate = ModelRun(
        run_id=candidate_id,
        task="forecast",
        algorithm="lightgbm",
        status="succeeded",
        is_champion=False,
        trained_at=now - datetime.timedelta(hours=2),
        metrics={"wape": 0.12, "gate_passed": True, "baseline_wape": 0.20},
        params={"lr": 0.05},
        git_sha="abcdef2",
        dataset_hash="hash2",
        seed=42,
        mlflow_run_id="mlflow-cand-2",
        mlflow_model_name="retailpulse-forecast",
        mlflow_model_version="2",
    )

    # 3. Candidate that failed gate (worse than champion)
    failed_gate_id = uuid.uuid4()
    failed_gate = ModelRun(
        run_id=failed_gate_id,
        task="forecast",
        algorithm="xgboost",
        status="succeeded",
        is_champion=False,
        trained_at=now - datetime.timedelta(hours=1),
        metrics={
            "wape": 0.35,
            "gate_passed": False,
            "gate_reason": "WAPE 0.35 is worse than champion WAPE 0.20",
        },
        params={"depth": 8},
        git_sha="abcdef3",
        dataset_hash="hash3",
        seed=42,
        mlflow_run_id="mlflow-fail-3",
        mlflow_model_name="retailpulse-forecast",
        mlflow_model_version="3",
    )

    # 4. Model run with failed training status
    failed_status_id = uuid.uuid4()
    failed_status = ModelRun(
        run_id=failed_status_id,
        task="forecast",
        algorithm="prophet",
        status="failed",
        is_champion=False,
        trained_at=now - datetime.timedelta(minutes=30),
        metrics={},
        params={},
        git_sha="abcdef4",
        dataset_hash="hash4",
        seed=42,
        mlflow_run_id="mlflow-fail-4",
    )

    db_session.add_all([champ, candidate, failed_gate, failed_status])
    db_session.commit()

    return {
        "admin_id": admin_id,
        "viewer_id": viewer_id,
        "analyst_id": analyst_id,
        "champion_id": champ_id,
        "candidate_id": candidate_id,
        "failed_gate_id": failed_gate_id,
        "failed_status_id": failed_status_id,
    }


# ==============================================================================
# Service Level Unit Tests: promote_model_run & rollback_model_run
# ==============================================================================


@pytest.mark.anyio
async def test_promote_model_run_success(
    db_session: Session,
    seeded_models: dict[str, Any],
    mock_pipeline_trigger: PipelineTrigger,
) -> None:
    """Verify promote_model_run flips is_champion, updates aliases, and triggers score_all."""
    candidate_id = seeded_models["candidate_id"]
    champ_id = seeded_models["champion_id"]
    admin_id = seeded_models["admin_id"]

    mock_client = MagicMock()
    mock_mv = MagicMock()
    mock_mv.version = "1"
    mock_client.get_model_version_by_alias.return_value = mock_mv

    with patch("retailpulse.pipelines.promotion.MlflowClient", return_value=mock_client):
        result = await promote_model_run(
            run_id=candidate_id,
            session=db_session,
            actor_id=admin_id,
            pipeline_trigger=mock_pipeline_trigger,
        )

    # Assert PromoteResult returned accurately
    assert isinstance(result, PromoteResult)
    assert result.task == "forecast"
    assert result.run_id == candidate_id
    assert result.previous_champion_run_id == champ_id
    assert result.champion_version == "2"
    assert result.is_rollback is False
    assert result.score_all_pipeline_run_id is not None

    # Assert database states
    db_session.expire_all()
    old_champ = db_session.get(ModelRun, champ_id)
    new_champ = db_session.get(ModelRun, candidate_id)
    assert old_champ is not None
    assert old_champ.is_champion is False
    assert new_champ is not None
    assert new_champ.is_champion is True

    # Assert MLflow alias updates called
    mock_client.set_registered_model_alias.assert_any_call(
        "retailpulse-forecast", "previous_champion", "1"
    )
    mock_client.set_registered_model_alias.assert_any_call(
        "retailpulse-forecast", "champion", "2"
    )

    # Assert PipelineRun created for score_all
    pipe_run = db_session.get(PipelineRun, result.score_all_pipeline_run_id)
    assert pipe_run is not None
    assert pipe_run.flow_name == "score_all"
    assert pipe_run.trigger == "promote"
    assert pipe_run.params["promoted_run_id"] == str(candidate_id)

    # Assert PipelineTrigger was invoked
    mock_pipeline_trigger.trigger.assert_awaited_once_with(  # type: ignore[attr-defined]
        flow_name="score_all",
        run_id=pipe_run.run_id,
        params=pipe_run.params,
    )

    # Assert AuditLog entry
    audit = db_session.execute(
        sa.select(AuditLog).where(
            AuditLog.action == "promote",
            AuditLog.resource == f"model:forecast:{candidate_id}",
        )
    ).scalar_one_or_none()
    assert audit is not None
    assert audit.user_id == admin_id
    assert audit.details["previous_champion_run_id"] == str(champ_id)
    assert audit.details["champion_version"] == "2"
    assert audit.details["is_rollback"] is False


@pytest.mark.anyio
async def test_promote_model_run_refuses_failed_gate(
    db_session: Session,
    seeded_models: dict[str, Any],
    mock_pipeline_trigger: PipelineTrigger,
) -> None:
    """Verify candidate failing evaluation gate is refused with ModelGateError."""
    failed_gate_id = seeded_models["failed_gate_id"]
    champ_id = seeded_models["champion_id"]

    with pytest.raises(ModelGateError) as exc_info:
        await promote_model_run(
            run_id=failed_gate_id,
            session=db_session,
            pipeline_trigger=mock_pipeline_trigger,
        )

    assert "did not pass the promotion gate" in str(exc_info.value)
    assert "WAPE 0.35 is worse than champion WAPE 0.20" in str(exc_info.value)

    # Champion must remain untouched
    db_session.expire_all()
    champ = db_session.get(ModelRun, champ_id)
    failed = db_session.get(ModelRun, failed_gate_id)
    assert champ is not None
    assert champ.is_champion is True
    assert failed is not None
    assert failed.is_champion is False

    # Pipeline trigger must NOT have been called
    mock_pipeline_trigger.trigger.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.anyio
async def test_promote_model_run_refuses_non_succeeded_status(
    db_session: Session,
    seeded_models: dict[str, Any],
) -> None:
    """Verify runs with status other than succeeded are refused with ConflictError."""
    failed_status_id = seeded_models["failed_status_id"]

    with pytest.raises(ConflictError) as exc_info:
        await promote_model_run(
            run_id=failed_status_id,
            session=db_session,
        )

    assert "status 'failed'" in str(exc_info.value)
    assert "only succeeded runs may be promoted" in str(exc_info.value)


@pytest.mark.anyio
async def test_promote_model_run_not_found(db_session: Session) -> None:
    """Verify non-existent run raises NotFoundError."""
    missing_id = uuid.uuid4()
    with pytest.raises(NotFoundError):
        await promote_model_run(run_id=missing_id, session=db_session)


@pytest.mark.anyio
async def test_rollback_model_run_success_via_alias(
    db_session: Session,
    seeded_models: dict[str, Any],
    mock_pipeline_trigger: PipelineTrigger,
) -> None:
    """Verify rollback finds previous champion via MLflow alias and restores it."""
    candidate_id = seeded_models["candidate_id"]
    champ_id = seeded_models["champion_id"]
    admin_id = seeded_models["admin_id"]

    # First, promote candidate to make it champion and champ_id the previous champion
    mock_client = MagicMock()
    mock_v1 = MagicMock(version="1", run_id="mlflow-champ-1")
    mock_client.get_model_version_by_alias.return_value = mock_v1

    with patch("retailpulse.pipelines.promotion.MlflowClient", return_value=mock_client):
        # 1. Promote candidate
        await promote_model_run(
            run_id=candidate_id,
            session=db_session,
            actor_id=admin_id,
            pipeline_trigger=mock_pipeline_trigger,
        )

        # 2. Now rollback to previous champion
        mock_pipeline_trigger.trigger.reset_mock()  # type: ignore[attr-defined]
        rollback_result = await rollback_model_run(
            task="forecast",
            session=db_session,
            actor_id=admin_id,
            pipeline_trigger=mock_pipeline_trigger,
        )

    assert rollback_result.is_rollback is True
    assert rollback_result.run_id == champ_id
    assert rollback_result.previous_champion_run_id == candidate_id

    # Verify DB state reverted
    db_session.expire_all()
    restored_champ = db_session.get(ModelRun, champ_id)
    demoted_cand = db_session.get(ModelRun, candidate_id)
    assert restored_champ is not None
    assert restored_champ.is_champion is True
    assert demoted_cand is not None
    assert demoted_cand.is_champion is False


@pytest.mark.anyio
async def test_rollback_model_run_refusal_when_no_previous_champion(
    db_session: Session,
) -> None:
    """Verify rollback raises ConflictError when no previous champion exists."""
    # Empty DB has no models
    with pytest.raises(ConflictError) as exc_info:
        await rollback_model_run(task="forecast", session=db_session)

    assert "No previous champion found" in str(exc_info.value)


@pytest.mark.anyio
async def test_rollback_model_run_invalid_task(db_session: Session) -> None:
    """Verify invalid task raises DomainValidationError."""
    with pytest.raises(DomainValidationError):
        await rollback_model_run(task="nonexistent_task", session=db_session)


# ==============================================================================
# API Endpoint Integration Tests: /promote and /rollback
# ==============================================================================


@pytest.fixture
def api_test_setup(db_engine: sa.Engine, seeded_models: dict[str, Any]):
    """Set up TestClient with seeded SQLite DB and auth headers."""
    app = create_app()

    def override_get_db():
        session_cls = sessionmaker(bind=db_engine)
        session = session_cls()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    admin_token = create_access_token(
        user_id=seeded_models["admin_id"],
        role="admin",
    )
    viewer_token = create_access_token(user_id=seeded_models["viewer_id"], role="viewer")
    analyst_token = create_access_token(user_id=seeded_models["analyst_id"], role="analyst")

    client = TestClient(app)

    yield {
        "client": client,
        "admin_headers": {"Authorization": f"Bearer {admin_token}"},
        "viewer_headers": {"Authorization": f"Bearer {viewer_token}"},
        "analyst_headers": {"Authorization": f"Bearer {analyst_token}"},
        "models": seeded_models,
    }

    app.dependency_overrides.clear()


def test_api_promote_gate_refusal_and_success(api_test_setup: dict[str, Any]) -> None:
    """Test API promote endpoint gate enforcement (409) and successful promotion (200)."""
    client: TestClient = api_test_setup["client"]
    admin = api_test_setup["admin_headers"]
    viewer = api_test_setup["viewer_headers"]
    analyst = api_test_setup["analyst_headers"]
    cand_id = api_test_setup["models"]["candidate_id"]
    failed_gate_id = api_test_setup["models"]["failed_gate_id"]

    # 1. Non-admin forbidden (403)
    assert client.post(f"/api/v1/models/{cand_id}/promote", headers=viewer).status_code == 403
    assert client.post(f"/api/v1/models/{cand_id}/promote", headers=analyst).status_code == 403

    # 2. Gate failure returns 409
    res_gate = client.post(f"/api/v1/models/{failed_gate_id}/promote", headers=admin)
    assert res_gate.status_code == 409
    body_gate = res_gate.json()
    assert "did not pass the promotion gate" in body_gate["detail"]

    # 3. Successful promotion returns 200
    with patch("retailpulse.pipelines.promotion.MlflowClient"):
        res_ok = client.post(f"/api/v1/models/{cand_id}/promote", headers=admin)
    assert res_ok.status_code == 200
    body_ok = res_ok.json()
    assert body_ok["task"] == "forecast"
    assert body_ok["run_id"] == str(cand_id)
    assert body_ok["previous_champion_run_id"] == str(api_test_setup["models"]["champion_id"])
    assert "score_all_pipeline_run_id" in body_ok
    assert body_ok["is_rollback"] is False


def test_api_rollback_endpoint(api_test_setup: dict[str, Any]) -> None:
    """Test API rollback endpoint RBAC, invalid task (422), and successful rollback (200)."""
    client: TestClient = api_test_setup["client"]
    admin = api_test_setup["admin_headers"]
    viewer = api_test_setup["viewer_headers"]
    cand_id = api_test_setup["models"]["candidate_id"]

    # 1. Non-admin forbidden (403)
    assert client.post("/api/v1/models/forecast/rollback", headers=viewer).status_code == 403

    # 2. Invalid task (422)
    res_inv = client.post("/api/v1/models/invalid_task/rollback", headers=admin)
    assert res_inv.status_code == 422

    with patch("retailpulse.pipelines.promotion.MlflowClient"):
        # First promote candidate to establish history
        client.post(f"/api/v1/models/{cand_id}/promote", headers=admin)

        # 3. Rollback endpoint returns 200 with is_rollback=True
        res_rb = client.post("/api/v1/models/forecast/rollback", headers=admin)
        assert res_rb.status_code == 200
        body_rb = res_rb.json()
        assert body_rb["task"] == "forecast"
        assert body_rb["run_id"] == str(api_test_setup["models"]["champion_id"])
        assert body_rb["is_rollback"] is True
