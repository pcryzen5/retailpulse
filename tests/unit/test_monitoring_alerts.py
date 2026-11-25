"""Unit tests for operational alerts, deduplication, and webhook dispatching.

Implements P7-06 verification per ARCHITECTURE §7, DATABASE.md §7, and SECURITY.md §8:
- Open-state deduplication on `dedupe_key`.
- Re-raising alerts once previously acknowledged.
- Webhook dispatch with operator-set URL, timeout, and disabled redirects (SEC-INP-11).
- Non-blocking webhook failure safety.
- Domain alert rule helpers (pipeline failure, quality rejection, drift, brute force, etc.).
"""

from __future__ import annotations

import datetime
import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.config import Settings
from retailpulse.db.base import Base
from retailpulse.db.models.ops import Alert
from retailpulse.errors import DomainValidationError
from retailpulse.monitoring.alerts import (
    KIND_CHURN_DEGRADED,
    KIND_DATA_QUALITY_FAILED,
    KIND_DRIFT_DETECTED,
    KIND_FAILED_LOGINS_SPIKE,
    KIND_FORECAST_DEGRADED,
    KIND_FORECAST_NO_MODEL_BEATS_BASELINE,
    KIND_PIPELINE_FAILED,
    KIND_STOCKOUT_SPIKE,
    create_alert,
    emit_churn_degraded_alert,
    emit_data_quality_alert,
    emit_drift_detected_alert,
    emit_failed_logins_spike_alert,
    emit_forecast_degraded_alert,
    emit_no_model_beats_baseline_alert,
    emit_pipeline_failure_alert,
    emit_stockout_spike_alert,
    send_alert_webhook,
)
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


# ==============================================================================
# 1. Alert Creation and Validation
# ==============================================================================


def test_create_alert_basic(db_session: Session) -> None:
    """Verify basic alert creation without deduplication key."""
    alert, is_new = create_alert(
        session=db_session,
        severity="warning",
        kind="test_kind",
        message="Test alert message",
        payload={"foo": "bar"},
        dispatch_webhook=False,
    )
    db_session.commit()

    assert is_new is True
    assert alert.alert_id is not None
    assert alert.severity == "warning"
    assert alert.kind == "test_kind"
    assert alert.message == "Test alert message"
    assert alert.payload == {"foo": "bar"}
    assert alert.acknowledged_at is None


def test_create_alert_invalid_severity_raises(db_session: Session) -> None:
    """Verify invalid severity raises DomainValidationError."""
    with pytest.raises(DomainValidationError) as exc_info:
        create_alert(
            session=db_session,
            severity="extreme_critical",
            kind="test",
            message="bad severity",
            dispatch_webhook=False,
        )
    assert "Invalid alert severity" in str(exc_info.value)


# ==============================================================================
# 2. Deduplication Tests (DATABASE.md §7)
# ==============================================================================


def test_alert_open_state_deduplication(db_session: Session) -> None:
    """Verify an identical open alert with the same dedupe_key is suppressed."""
    dedupe_key = "pipeline_failed:etl:run-123"

    # 1. Initial creation
    alert1, is_new1 = create_alert(
        session=db_session,
        severity="critical",
        kind=KIND_PIPELINE_FAILED,
        message="Initial pipeline failure",
        dedupe_key=dedupe_key,
        dispatch_webhook=False,
    )
    db_session.commit()
    assert is_new1 is True

    # 2. Duplicate while unacknowledged -> suppressed
    alert2, is_new2 = create_alert(
        session=db_session,
        severity="critical",
        kind=KIND_PIPELINE_FAILED,
        message="Repeated pipeline failure",
        dedupe_key=dedupe_key,
        dispatch_webhook=False,
    )
    db_session.commit()
    assert is_new2 is False
    assert alert2.alert_id == alert1.alert_id

    # Verify only 1 row exists in ops.alert
    total = db_session.scalar(sa.select(sa.func.count(Alert.alert_id)))
    assert total == 1


def test_alert_dedupe_allows_new_after_acknowledgement(db_session: Session) -> None:
    """Verify that once an alert is acknowledged, the dedupe_key can trigger a new alert."""
    dedupe_key = "drift_detected:forecast:rep-456"

    # 1. Create first alert
    alert1, is_new1 = create_alert(
        session=db_session,
        severity="warning",
        kind=KIND_DRIFT_DETECTED,
        message="Drift batch 1",
        dedupe_key=dedupe_key,
        dispatch_webhook=False,
    )
    db_session.commit()
    assert is_new1 is True

    # 2. Acknowledge first alert
    alert1.acknowledged_at = utc_now()
    db_session.commit()

    # 3. Trigger again with same dedupe_key -> new alert created
    alert2, is_new2 = create_alert(
        session=db_session,
        severity="warning",
        kind=KIND_DRIFT_DETECTED,
        message="Drift batch 2",
        dedupe_key=dedupe_key,
        dispatch_webhook=False,
    )
    db_session.commit()
    assert is_new2 is True
    assert alert2.alert_id != alert1.alert_id

    # Verify 2 rows in database (1 acknowledged, 1 open)
    total = db_session.scalar(sa.select(sa.func.count(Alert.alert_id)))
    assert total == 2


# ==============================================================================
# 3. Webhook Tests (SEC-INP-11)
# ==============================================================================


@patch("retailpulse.monitoring.alerts.httpx.Client")
def test_send_alert_webhook_success(mock_client_cls: MagicMock, db_session: Session) -> None:
    """Verify webhook sends POST with payload, timeout, and redirects disabled."""
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_client.post.return_value = mock_resp
    mock_client_cls.return_value.__enter__.return_value = mock_client

    alert = Alert(
        alert_id=42,
        severity="critical",
        kind=KIND_PIPELINE_FAILED,
        message="Pipeline crashed",
        payload={"flow": "etl"},
        dedupe_key="pipe_1",
        created_at=utc_now(),
    )

    success = send_alert_webhook(
        alert,
        webhook_url="https://hooks.example.com/alerts",
        timeout_seconds=3.0,
    )

    assert success is True
    # Verify client initialization: follow_redirects=False (SEC-INP-11), timeout enforced
    mock_client_cls.assert_called_once_with(
        timeout=3.0,
        follow_redirects=False,
    )

    # Verify POST payload
    mock_client.post.assert_called_once()
    call_args = mock_client.post.call_args
    assert call_args[0][0] == "https://hooks.example.com/alerts"
    payload = call_args[1]["json"]
    assert payload["alert_id"] == 42
    assert payload["severity"] == "critical"
    assert payload["kind"] == KIND_PIPELINE_FAILED
    assert payload["message"] == "Pipeline crashed"
    assert payload["payload"] == {"flow": "etl"}
    assert payload["dedupe_key"] == "pipe_1"


def test_send_alert_webhook_no_op_when_no_url() -> None:
    """Verify webhook returns False immediately if no webhook URL is configured."""
    alert = Alert(
        alert_id=1,
        severity="info",
        kind="test",
        message="msg",
    )
    with patch("retailpulse.monitoring.alerts.get_settings") as mock_settings:
        mock_settings.return_value = Settings(alert_webhook_url=None)
        res = send_alert_webhook(alert, webhook_url=None)
        assert res is False


@patch("retailpulse.monitoring.alerts.httpx.Client")
def test_webhook_failure_does_not_abort_alert_creation(
    mock_client_cls: MagicMock,
    db_session: Session,
) -> None:
    """Verify that webhook network exceptions or timeouts do not fail create_alert."""
    mock_client = MagicMock()
    mock_client.post.side_effect = httpx.ConnectTimeout("Connection timed out")
    mock_client_cls.return_value.__enter__.return_value = mock_client

    alert, is_new = create_alert(
        session=db_session,
        severity="critical",
        kind=KIND_PIPELINE_FAILED,
        message="Database should commit despite webhook timeout",
        dispatch_webhook=True,
        webhook_url="https://unreachable.local/webhook",
    )
    db_session.commit()

    assert is_new is True
    assert alert.alert_id is not None
    # Verify alert exists in database
    row = db_session.execute(
        sa.select(Alert).where(Alert.alert_id == alert.alert_id)
    ).scalar_one_or_none()
    assert row is not None


# ==============================================================================
# 4. Domain Alert Rule Helpers Tests
# ==============================================================================


def test_domain_emit_pipeline_failure_alert(db_session: Session) -> None:
    """Test emit_pipeline_failure_alert helper."""
    run_id = uuid.uuid4()
    alert, is_new = emit_pipeline_failure_alert(
        session=db_session,
        flow_name="etl",
        run_id=run_id,
        error_message="Connection lost",
        details={"step": "clean"},
        dispatch_webhook=False,
    )
    assert is_new is True
    assert alert.severity == "critical"
    assert alert.kind == KIND_PIPELINE_FAILED
    assert "etl" in alert.message
    assert alert.payload["pipeline_run_id"] == str(run_id)


def test_domain_emit_data_quality_alert(db_session: Session) -> None:
    """Test emit_data_quality_alert helper."""
    batch_id = uuid.uuid4()
    alert, is_new = emit_data_quality_alert(
        session=db_session,
        batch_id=batch_id,
        reasons=["Exceeded rejection threshold (5.2% > 5%)"],
        quarantined_count=52,
        total_count=1000,
        dispatch_webhook=False,
    )
    assert is_new is True
    assert alert.severity == "critical"
    assert alert.kind == KIND_DATA_QUALITY_FAILED
    assert "rejected" in alert.message
    assert alert.payload["quarantined_count"] == 52


def test_domain_emit_drift_detected_alert(db_session: Session) -> None:
    """Test emit_drift_detected_alert helper."""
    rep_id = uuid.uuid4()
    alert, is_new = emit_drift_detected_alert(
        session=db_session,
        task="forecast",
        share_drifted=0.67,
        drifted_features=["quantity", "unit_price"],
        report_id=rep_id,
        dispatch_webhook=False,
    )
    assert is_new is True
    assert alert.severity == "warning"
    assert alert.kind == KIND_DRIFT_DETECTED
    assert "67.0%" in alert.message
    assert alert.payload["report_id"] == str(rep_id)


def test_domain_emit_no_model_beats_baseline_alert(db_session: Session) -> None:
    """Test emit_no_model_beats_baseline_alert helper (TRD §5.1)."""
    alert, is_new = emit_no_model_beats_baseline_alert(
        session=db_session,
        task="forecast",
        baseline_wape=0.35,
        candidate_wapes=[0.42, 0.38],
        dispatch_webhook=False,
    )
    assert is_new is True
    assert alert.severity == "warning"
    assert alert.kind == KIND_FORECAST_NO_MODEL_BEATS_BASELINE
    assert "seasonal naive baseline" in alert.message
    assert alert.payload["baseline_wape"] == 0.35


def test_domain_emit_failed_logins_spike_alert(db_session: Session) -> None:
    """Test emit_failed_logins_spike_alert helper (SEC-LOG-03)."""
    alert, is_new = emit_failed_logins_spike_alert(
        session=db_session,
        ip_address="198.51.100.24",
        failed_count=12,
        window_minutes=10,
        dispatch_webhook=False,
    )
    assert is_new is True
    assert alert.severity == "warning"
    assert alert.kind == KIND_FAILED_LOGINS_SPIKE
    assert "198.51.100.24" in alert.message
    assert alert.payload["failed_count"] == 12


def test_domain_emit_stockout_spike_alert(db_session: Session) -> None:
    """Test emit_stockout_spike_alert helper."""
    alert, is_new = emit_stockout_spike_alert(
        session=db_session,
        reorder_now_count=45,
        total_skus=100,
        dispatch_webhook=False,
    )
    assert is_new is True
    assert alert.severity == "warning"
    assert alert.kind == KIND_STOCKOUT_SPIKE
    assert "45.0%" in alert.message
    assert alert.payload["reorder_now_count"] == 45


def test_domain_emit_model_degraded_alerts(db_session: Session) -> None:
    """Test emit_forecast_degraded_alert and emit_churn_degraded_alert."""
    run_id = uuid.uuid4()
    d1 = datetime.date(2026, 11, 1)
    d2 = datetime.date(2026, 11, 15)

    # Forecast degraded
    a_fc, is_fc = emit_forecast_degraded_alert(
        session=db_session,
        model_run_id=run_id,
        realized_wape=0.45,
        backtest_wape=0.20,
        baseline_wape=0.35,
        window_start=d1,
        window_end=d2,
        reason="Realized WAPE exceeded backtest multiplier",
        dispatch_webhook=False,
    )
    assert is_fc is True
    assert a_fc.kind == KIND_FORECAST_DEGRADED

    # Churn degraded
    a_ch, is_ch = emit_churn_degraded_alert(
        session=db_session,
        model_run_id=run_id,
        realized_pr_auc=0.22,
        backtest_pr_auc=0.65,
        baseline_pr_auc=0.25,
        window_start=d1,
        window_end=d2,
        reason="Realized PR-AUC dropped below baseline",
        dispatch_webhook=False,
    )
    assert is_ch is True
    assert a_ch.kind == KIND_CHURN_DEGRADED
