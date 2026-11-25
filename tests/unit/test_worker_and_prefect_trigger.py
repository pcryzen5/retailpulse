"""Unit tests for Prefect worker deployments, schedules, and PrefectTrigger."""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from retailpulse.api.app import create_app
from retailpulse.config import Settings
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.ops import PipelineRun
from retailpulse.pipelines.flows import execute_pipeline_flow
from retailpulse.pipelines.trigger import PrefectTrigger
from retailpulse.pipelines.worker import get_worker_deployments
from retailpulse.security.jwt import create_access_token


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_: Any, compiler: Any, **kw: Any) -> str:
    """Compile PostgreSQL CITEXT columns to standard TEXT on SQLite."""
    return "TEXT"


@pytest.fixture
def test_db_session():
    """Create an isolated in-memory SQLite database for pipeline trigger testing."""
    translate_map = {"app": None, "core": None, "ops": None, "ml": None}
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        execution_options={"schema_translate_map": translate_map},
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    with patch("retailpulse.db.session.get_engine", return_value=engine), patch(
        "retailpulse.db.session.get_session_factory", return_value=session_factory
    ), session_factory() as session:
        yield session

    Base.metadata.drop_all(bind=engine)


def test_get_worker_deployments_schedules_and_names():
    """Verify worker deployments define all 6 workflows with ARCHITECTURE §6 schedules."""
    deployments = get_worker_deployments()
    assert len(deployments) == 6

    dep_map = {d.name: d for d in deployments}
    expected_names = {
        "etl-deployment",
        "score_all-deployment",
        "train_forecast-deployment",
        "train_segmentation-deployment",
        "train_churn-deployment",
        "drift_check-deployment",
    }
    assert set(dep_map.keys()) == expected_names

    # etl: daily 01:00 UTC ('0 1 * * *')
    etl_dep = dep_map["etl-deployment"]
    assert len(etl_dep.schedules) == 1
    assert etl_dep.schedules[0].schedule.cron == "0 1 * * *"
    assert etl_dep.schedules[0].schedule.timezone == "UTC"

    # score_all: chained / manual (no periodic schedule)
    score_dep = dep_map["score_all-deployment"]
    assert len(score_dep.schedules) == 0

    # train_forecast: Sunday 02:00 UTC ('0 2 * * 0')
    tf_dep = dep_map["train_forecast-deployment"]
    assert len(tf_dep.schedules) == 1
    assert tf_dep.schedules[0].schedule.cron == "0 2 * * 0"
    assert tf_dep.schedules[0].schedule.timezone == "UTC"

    # train_segmentation: Sunday 03:00 UTC ('0 3 * * 0')
    ts_dep = dep_map["train_segmentation-deployment"]
    assert len(ts_dep.schedules) == 1
    assert ts_dep.schedules[0].schedule.cron == "0 3 * * 0"
    assert ts_dep.schedules[0].schedule.timezone == "UTC"

    # train_churn: Sunday 03:30 UTC ('30 3 * * 0')
    tc_dep = dep_map["train_churn-deployment"]
    assert len(tc_dep.schedules) == 1
    assert tc_dep.schedules[0].schedule.cron == "30 3 * * 0"
    assert tc_dep.schedules[0].schedule.timezone == "UTC"

    # drift_check: daily 04:00 UTC ('0 4 * * *')
    dc_dep = dep_map["drift_check-deployment"]
    assert len(dc_dep.schedules) == 1
    assert dc_dep.schedules[0].schedule.cron == "0 4 * * *"
    assert dc_dep.schedules[0].schedule.timezone == "UTC"


@pytest.mark.anyio
async def test_prefect_trigger_success(test_db_session):
    """Test PrefectTrigger successfully creates a flow run and records prefect_flow_run_id."""
    run_id = uuid.uuid4()
    pipeline_run = PipelineRun(
        run_id=run_id,
        flow_name="etl",
        trigger="manual",
        status="queued",
        params={"test_param": 123},
    )
    test_db_session.add(pipeline_run)
    test_db_session.commit()

    mock_deployment = MagicMock()
    mock_deployment.id = uuid.uuid4()

    mock_flow_run = MagicMock()
    mock_flow_run.id = uuid.uuid4()
    mock_flow_run.name = "etl-flow-run-1"

    mock_client = AsyncMock()
    mock_client.read_deployment_by_name = AsyncMock(return_value=mock_deployment)
    mock_client.create_flow_run_from_deployment = AsyncMock(return_value=mock_flow_run)

    client_context_mock = AsyncMock()
    client_context_mock.__aenter__.return_value = mock_client
    client_context_mock.__aexit__.return_value = None

    trigger = PrefectTrigger(api_url="http://test-prefect:4200/api")

    with patch("retailpulse.pipelines.trigger.PrefectClient", return_value=client_context_mock):
        result = await trigger.trigger(
            flow_name="etl",
            run_id=run_id,
            params={"test_param": 123},
        )

    assert result is True
    mock_client.read_deployment_by_name.assert_awaited_once_with("etl/etl-deployment")
    mock_client.create_flow_run_from_deployment.assert_awaited_once_with(
        deployment_id=mock_deployment.id,
        parameters={"run_id": str(run_id), "params": {"test_param": 123}},
    )

    # Verify prefect_flow_run_id was written to the DB
    test_db_session.expire_all()
    updated_run = test_db_session.get(PipelineRun, run_id)
    assert updated_run is not None
    assert updated_run.prefect_flow_run_id == str(mock_flow_run.id)


@pytest.mark.anyio
async def test_prefect_trigger_failure_raises(test_db_session):
    """Test PrefectTrigger raises an exception when Prefect server is unreachable."""
    run_id = uuid.uuid4()
    pipeline_run = PipelineRun(
        run_id=run_id,
        flow_name="score_all",
        trigger="manual",
        status="queued",
        params={},
    )
    test_db_session.add(pipeline_run)
    test_db_session.commit()

    mock_client = AsyncMock()
    mock_client.read_deployment_by_name = AsyncMock(
        side_effect=ConnectionError("Failed to reach Prefect server")
    )

    client_context_mock = AsyncMock()
    client_context_mock.__aenter__.return_value = mock_client
    client_context_mock.__aexit__.return_value = None

    trigger = PrefectTrigger(api_url="http://test-prefect:4200/api")

    with (
        patch("retailpulse.pipelines.trigger.PrefectClient", return_value=client_context_mock),
        pytest.raises(ConnectionError, match="Failed to reach Prefect server"),
    ):
        await trigger.trigger(
            flow_name="score_all",
            run_id=run_id,
            params={},
        )


def test_api_trigger_with_prefect_trigger_end_to_end(test_db_session):
    """Test API POST trigger endpoint dispatches via PrefectTrigger."""
    admin_id = uuid.uuid4()
    admin_user = AppUser(
        user_id=admin_id,
        email="admin@retailpulse.dev",
        role="admin",
        password_hash="test_password_hash",
        is_active=True,
    )
    test_db_session.add(admin_user)
    test_db_session.commit()

    token = create_access_token(admin_id, role="admin")
    headers = {"Authorization": f"Bearer {token}"}

    mock_deployment = MagicMock()
    mock_deployment.id = uuid.uuid4()

    mock_flow_run = MagicMock()
    mock_flow_run.id = uuid.uuid4()
    mock_flow_run.name = "etl-flow-run-abc"

    mock_client = AsyncMock()
    mock_client.read_deployment_by_name = AsyncMock(return_value=mock_deployment)
    mock_client.create_flow_run_from_deployment = AsyncMock(return_value=mock_flow_run)

    client_context_mock = AsyncMock()
    client_context_mock.__aenter__.return_value = mock_client
    client_context_mock.__aexit__.return_value = None

    trigger = PrefectTrigger(api_url="http://prefect:4200/api")
    app = create_app(settings=Settings(env="test"), pipeline_trigger=trigger)

    with (
        patch("retailpulse.pipelines.trigger.PrefectClient", return_value=client_context_mock),
        TestClient(app) as client,
    ):
        resp = client.post(
            "/api/v1/pipelines/etl/trigger",
            json={"params": {"full_refresh": False}},
            headers=headers,
        )

    assert resp.status_code == 202
    data = resp.json()
    assert data["flow_name"] == "etl"
    assert data["status"] == "queued"
    assert "pipeline_run_id" in data

    # Verify run record exists in DB
    created_id = uuid.UUID(data["pipeline_run_id"])
    run_rec = test_db_session.get(PipelineRun, created_id)
    assert run_rec is not None
    assert run_rec.flow_name == "etl"
    assert run_rec.status == "queued"


def test_execute_pipeline_flow_auto_creates_run_when_run_id_is_none(test_db_session):
    """Verify execute_pipeline_flow generates a scheduled PipelineRun when run_id is None."""
    with patch(
        "retailpulse.pipelines.flows.FLOW_DISPATCH_TABLE",
        {"drift_check": MagicMock(return_value={"drift": False})},
    ):
        result = execute_pipeline_flow.fn("drift_check", run_id=None, params={"mode": "fast"})

    assert result == {"drift": False}
    # Check that a run was created in ops.pipeline_run
    runs = test_db_session.query(PipelineRun).filter_by(flow_name="drift_check").all()
    assert len(runs) == 1
    assert runs[0].trigger == "schedule"
    assert runs[0].status == "succeeded"


def test_docker_compose_configuration():
    """Verify docker-compose.yml defines all required services and Prefect 3 wiring."""
    with open("docker-compose.yml", encoding="utf-8") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "postgres" in services
    assert "mlflow" in services
    assert "prefect" in services
    assert "worker" in services
    assert "api" in services

    # Verify prefect image is version 3
    prefect_svc = services["prefect"]
    assert "prefecthq/prefect:3" in prefect_svc["image"]

    # Verify worker depends on postgres, mlflow, and prefect
    worker_svc = services["worker"]
    assert "postgres" in worker_svc["depends_on"]
    assert "mlflow" in worker_svc["depends_on"]
    assert "prefect" in worker_svc["depends_on"]
    assert worker_svc["environment"]["PREFECT_API_URL"] == "http://prefect:4200/api"

    # Verify api depends on postgres and prefect, with RP_PIPELINE_TRIGGER=prefect
    api_svc = services["api"]
    assert "postgres" in api_svc["depends_on"]
    assert "prefect" in api_svc["depends_on"]
    assert api_svc["environment"]["RP_PIPELINE_TRIGGER"] == "prefect"
    assert api_svc["environment"]["PREFECT_API_URL"] == "http://prefect:4200/api"
