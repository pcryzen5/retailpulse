"""Unit tests for orchestration workflows and pipeline flows.

Verifies P7-01:
- State bookkeeping in `ops.pipeline_run`: status starts as 'running' and reaches 'succeeded'.
- Workflows call underlying domain functions (`etl`, `score_all`, `train_*`, `drift_check`).
- Failure recovery & alerts:
  * Failure marks the run as 'failed' and records the error trace.
  * Emits a critical alert in `ops.alert`.
- Retries with backoff handle transient failures.
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.db.base import Base
from retailpulse.db.models.ops import Alert, PipelineRun
from retailpulse.pipelines.flows import (
    FLOW_DISPATCH_TABLE,
    _execute_with_retry,
    execute_pipeline_flow,
)


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):  # type: ignore[no-untyped-def]
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def mock_db_session(monkeypatch: pytest.MonkeyPatch) -> Session:
    """In-memory SQLite session with table schema for pipeline run testing."""
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

    monkeypatch.setattr("retailpulse.pipelines.flows.session_scope", _test_session_scope)
    return session


def _create_initial_run(session: Session, flow_name: str) -> uuid.UUID:
    """Insert a queued pipeline run record."""
    run_id = uuid.uuid4()
    run = PipelineRun(
        run_id=run_id,
        flow_name=flow_name,
        trigger="manual",
        status="queued",
        params={},
    )
    session.add(run)
    session.commit()
    return run_id


def test_execute_pipeline_flow_success_bookkeeping(
    mock_db_session: Session,
) -> None:
    """Successful flow execution updates ops.pipeline_run status to 'succeeded'."""
    run_id = _create_initial_run(mock_db_session, "drift_check")

    res = execute_pipeline_flow.fn("drift_check", run_id=run_id)
    assert res is not None
    assert "report_id" in res

    # Verify database state
    run = mock_db_session.get(PipelineRun, run_id)
    assert run is not None
    assert run.status == "succeeded"
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.error is None


def test_execute_pipeline_flow_failure_marks_failed_and_creates_alert(
    mock_db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a flow runner fails, status becomes 'failed' and an alert is created."""
    run_id = _create_initial_run(mock_db_session, "score_all")

    # Force a failure in score_all
    def _failing_score(run_id: Any, params: Any) -> Any:
        raise RuntimeError("Model artifact storage unavailable")

    monkeypatch.setitem(FLOW_DISPATCH_TABLE, "score_all", _failing_score)

    with pytest.raises(RuntimeError, match="Model artifact storage unavailable"):
        execute_pipeline_flow.fn("score_all", run_id=run_id)

    # 1. Verify pipeline run status
    run = mock_db_session.get(PipelineRun, run_id)
    assert run is not None
    assert run.status == "failed"
    assert run.error is not None
    assert "Model artifact storage unavailable" in run.error

    # 2. Verify alert creation in ops.alert
    alerts = mock_db_session.query(Alert).all()
    assert len(alerts) >= 1
    pipe_alert = alerts[0]
    assert pipe_alert.severity == "critical"
    assert pipe_alert.kind == "pipeline_failure"
    assert "score_all" in pipe_alert.message
    assert str(run_id) in str(pipe_alert.payload)


def test_execute_pipeline_flow_dispatches_underlying_score_all(
    mock_db_session: Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that score_all coordinates forecast, segmentation, churn, and inventory."""
    run_id = _create_initial_run(mock_db_session, "score_all")

    mock_fc = MagicMock(return_value=MagicMock(predictions=[1, 2], model_run_id=uuid.uuid4()))
    mock_seg = MagicMock(return_value=MagicMock(customers_scored=100, model_run_id=uuid.uuid4()))
    mock_churn = MagicMock(return_value=MagicMock(predictions=[1, 2, 3], model_run_id=uuid.uuid4()))
    mock_inv = MagicMock(return_value=MagicMock(recommendations_count=50, snapshot_id=12))

    monkeypatch.setattr("retailpulse.pipelines.flows.fc_pred.predict_batch_forecast", mock_fc)
    monkeypatch.setattr(
        "retailpulse.pipelines.flows.seg_score.score_segmentation_pipeline", mock_seg
    )
    monkeypatch.setattr("retailpulse.pipelines.flows.churn_pred.score_churn_pipeline", mock_churn)
    monkeypatch.setattr(
        "retailpulse.pipelines.flows.inv_rec.generate_inventory_recommendations", mock_inv
    )

    res = execute_pipeline_flow.fn("score_all", run_id=run_id)

    assert mock_fc.called
    assert mock_seg.called
    assert mock_churn.called
    assert mock_inv.called

    assert "forecast" in res
    assert "segmentation" in res
    assert "churn" in res
    assert "inventory" in res


def test_retry_helper_recovers_after_transient_failure() -> None:
    """_execute_with_retry succeeds after transient errors within max_retries."""
    call_count = 0

    def _flaky_operation() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ConnectionError("Temporary connection drop")
        return "success"

    result = _execute_with_retry(
        _flaky_operation,
        max_retries=3,
        base_delay=0.01,
        step_name="test_flaky",
    )
    assert result == "success"
    assert call_count == 3


def test_retry_helper_raises_when_retries_exhausted() -> None:
    """_execute_with_retry raises exception when all attempts fail."""
    call_count = 0

    def _always_fails() -> None:
        nonlocal call_count
        call_count += 1
        raise ValueError("Permanent defect")

    with pytest.raises(ValueError, match="Permanent defect"):
        _execute_with_retry(
            _always_fails,
            max_retries=2,
            base_delay=0.01,
            step_name="test_failing",
        )
    assert call_count == 2
