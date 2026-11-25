"""Operational alert rules, deduplication, and webhook dispatching.

Implements P7-06, ARCHITECTURE §7, DATABASE.md §7, and SECURITY.md §8:
- Alert creation with open-state deduplication on `dedupe_key`.
- Security-enforced webhook notification (operator-set URL, timeout, redirects disabled).
- Domain alert rules for pipelines, data quality, drift, model degradation, and security anomalies.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

import httpx
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.ops import Alert
from retailpulse.errors import DomainValidationError
from retailpulse.logging import get_logger
from retailpulse.timeutil import utc_now

logger = get_logger(__name__)

VALID_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})

# Alert kinds per DATABASE.md §7 and TRD §6
KIND_PIPELINE_FAILED: str = "pipeline_failure"
KIND_DATA_QUALITY_FAILED: str = "data_quality_failed"
KIND_DRIFT_DETECTED: str = "drift_detected"
KIND_FORECAST_DEGRADED: str = "forecast_degraded"
KIND_FORECAST_NO_MODEL_BEATS_BASELINE: str = "forecast_no_model_beats_baseline"
KIND_FORECAST_COVERAGE_WARNING: str = "forecast_coverage_warning"
KIND_CHURN_DEGRADED: str = "churn_degraded"
KIND_STOCKOUT_SPIKE: str = "stockout_spike"
KIND_FAILED_LOGINS_SPIKE: str = "failed_logins_spike"

DEFAULT_WEBHOOK_TIMEOUT_SECONDS: float = 5.0


def send_alert_webhook(
    alert: Alert,
    *,
    webhook_url: str | None = None,
    timeout_seconds: float = DEFAULT_WEBHOOK_TIMEOUT_SECONDS,
) -> bool:
    """Dispatch an alert notification to an operator-configured webhook endpoint.

    Security constraints (SEC-INP-11):
    - Webhook URL is operator-configured via environment only.
    - Redirects are strictly disabled (follow_redirects=False).
    - Fixed network timeout applied.
    - Non-blocking: errors or network timeouts are logged but never fail the caller.
    """
    target_url = webhook_url
    if target_url is None:
        settings = get_settings()
        if settings.alert_webhook_url is not None:
            target_url = settings.alert_webhook_url.get_secret_value()

    if not target_url or not target_url.strip():
        return False

    target_url = target_url.strip()
    payload = {
        "alert_id": alert.alert_id,
        "severity": alert.severity,
        "kind": alert.kind,
        "message": alert.message,
        "payload": alert.payload or {},
        "dedupe_key": alert.dedupe_key,
        "created_at": alert.created_at.isoformat() if alert.created_at else None,
    }

    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
        ) as client:
            resp = client.post(target_url, json=payload)
            if resp.status_code >= 400:
                logger.warning(
                    "Alert webhook returned non-success HTTP status",
                    status_code=resp.status_code,
                    alert_id=alert.alert_id,
                    target_url=target_url,
                )
                return False

            logger.info(
                "Alert webhook dispatched successfully",
                alert_id=alert.alert_id,
                status_code=resp.status_code,
            )
            return True
    except Exception as exc:
        logger.warning(
            "Failed to dispatch alert webhook",
            alert_id=alert.alert_id,
            error=str(exc),
            target_url=target_url,
        )
        return False


def create_alert(
    session: Session,
    *,
    severity: str,
    kind: str,
    message: str,
    payload: dict[str, Any] | None = None,
    dedupe_key: str | None = None,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Create an operational alert with open-state deduplication.

    Per DATABASE.md §7:
    - `CREATE UNIQUE INDEX uq_alert_open_dedupe ON ops.alert (dedupe_key)
      WHERE acknowledged_at IS NULL AND dedupe_key IS NOT NULL;`
    - If an unacknowledged alert with the same `dedupe_key` exists, returns (existing_alert, False).
    - If no open duplicate exists, creates the alert, flushes, optionally sends webhook,
      and returns (new_alert, True).
    """
    clean_severity = severity.strip().lower()
    if clean_severity not in VALID_SEVERITIES:
        raise DomainValidationError(
            f"Invalid alert severity '{severity}'. Expected one of {sorted(VALID_SEVERITIES)}",
            details={"severity": severity, "valid_severities": sorted(VALID_SEVERITIES)},
        )

    clean_kind = kind.strip()
    clean_message = message.strip()

    # Deduplication check against open alerts
    if dedupe_key is not None and dedupe_key.strip():
        clean_dedupe_key = dedupe_key.strip()
        existing = session.execute(
            sa.select(Alert).where(
                Alert.dedupe_key == clean_dedupe_key,
                Alert.acknowledged_at.is_(None),
            )
        ).scalar_one_or_none()

        if existing is not None:
            logger.info(
                "Alert suppressed due to active unacknowledged duplicate",
                alert_id=existing.alert_id,
                dedupe_key=clean_dedupe_key,
                kind=existing.kind,
            )
            return existing, False
    else:
        clean_dedupe_key = None

    alert = Alert(
        severity=clean_severity,
        kind=clean_kind,
        message=clean_message,
        payload=payload or {},
        dedupe_key=clean_dedupe_key,
    )
    session.add(alert)
    session.flush()

    logger.info(
        "Alert created",
        alert_id=alert.alert_id,
        severity=alert.severity,
        kind=alert.kind,
        dedupe_key=alert.dedupe_key,
    )

    if dispatch_webhook:
        send_alert_webhook(alert, webhook_url=webhook_url)

    return alert, True


# ==============================================================================
# Domain Alert Rule Helpers
# ==============================================================================


def emit_pipeline_failure_alert(
    session: Session,
    *,
    flow_name: str,
    run_id: uuid.UUID,
    error_message: str,
    details: dict[str, Any] | None = None,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit a critical operational alert when a workflow pipeline run fails (ARCHITECTURE §6)."""
    return create_alert(
        session=session,
        severity="critical",
        kind=KIND_PIPELINE_FAILED,
        message=f"Pipeline flow '{flow_name}' failed: {error_message}",
        payload={
            "flow_name": flow_name,
            "pipeline_run_id": str(run_id),
            "error": error_message,
            "details": details or {},
        },
        dedupe_key=f"pipeline_failed:{flow_name}:{run_id}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_data_quality_alert(
    session: Session,
    *,
    batch_id: uuid.UUID,
    reasons: list[str],
    quarantined_count: int,
    total_count: int,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when ingestion data quality gates fail (Rule 8 rejection)."""
    reason_summary = "; ".join(reasons)
    msg = (
        f"Ingestion batch {batch_id} rejected due to quality check failures: {reason_summary}"
    )
    return create_alert(
        session=session,
        severity="critical",
        kind=KIND_DATA_QUALITY_FAILED,
        message=msg,
        payload={
            "batch_id": str(batch_id),
            "reasons": reasons,
            "quarantined_count": quarantined_count,
            "total_count": total_count,
        },
        dedupe_key=f"data_quality_failed:{batch_id}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_drift_detected_alert(
    session: Session,
    *,
    task: str,
    share_drifted: float,
    drifted_features: list[str],
    report_id: uuid.UUID,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when feature or prediction drift exceeds configured thresholds."""
    return create_alert(
        session=session,
        severity="warning",
        kind=KIND_DRIFT_DETECTED,
        message=(
            f"Statistically significant {task} drift detected in {len(drifted_features)} features "
            f"({share_drifted:.1%} drifted)"
        ),
        payload={
            "task": task,
            "share_drifted": round(share_drifted, 4),
            "drifted_features": drifted_features,
            "report_id": str(report_id),
        },
        dedupe_key=f"drift_detected:{task}:{report_id}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_forecast_degraded_alert(
    session: Session,
    *,
    model_run_id: uuid.UUID,
    realized_wape: float,
    backtest_wape: float,
    baseline_wape: float | None,
    window_start: datetime.date,
    window_end: datetime.date,
    reason: str,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when realized forecast error exceeds backtest or baseline (TRD §6)."""
    return create_alert(
        session=session,
        severity="warning",
        kind=KIND_FORECAST_DEGRADED,
        message=reason,
        payload={
            "model_run_id": str(model_run_id),
            "realized_wape": round(realized_wape, 4),
            "backtest_wape": round(backtest_wape, 4),
            "baseline_wape": round(baseline_wape, 4) if baseline_wape is not None else None,
            "window_start": str(window_start),
            "window_end": str(window_end),
        },
        dedupe_key=f"forecast_degraded:{model_run_id}:{window_start}:{window_end}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_churn_degraded_alert(
    session: Session,
    *,
    model_run_id: uuid.UUID,
    realized_pr_auc: float,
    backtest_pr_auc: float,
    baseline_pr_auc: float | None,
    window_start: datetime.date,
    window_end: datetime.date,
    reason: str,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when realized customer churn PR-AUC degrades (TRD §6)."""
    return create_alert(
        session=session,
        severity="warning",
        kind=KIND_CHURN_DEGRADED,
        message=reason,
        payload={
            "model_run_id": str(model_run_id),
            "realized_pr_auc": round(realized_pr_auc, 4),
            "backtest_pr_auc": round(backtest_pr_auc, 4),
            "baseline_pr_auc": round(baseline_pr_auc, 4) if baseline_pr_auc is not None else None,
            "window_start": str(window_start),
            "window_end": str(window_end),
        },
        dedupe_key=f"churn_degraded:{model_run_id}:{window_start}:{window_end}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_no_model_beats_baseline_alert(
    session: Session,
    *,
    task: str = "forecast",
    baseline_wape: float,
    candidate_wapes: list[float] | None = None,
    as_of_date: datetime.date | None = None,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when no trained model beats the seasonal naive baseline (TRD §5.1, §6)."""
    today_str = (as_of_date or utc_now().date()).isoformat()
    msg = (
        f"No candidate model beat the {task} seasonal naive baseline "
        f"(WAPE {baseline_wape:.4f}). Previous champion or baseline remains in production."
    )
    return create_alert(
        session=session,
        severity="warning",
        kind=KIND_FORECAST_NO_MODEL_BEATS_BASELINE,
        message=msg,
        payload={
            "task": task,
            "baseline_wape": round(baseline_wape, 4),
            "candidate_wapes": [round(w, 4) for w in (candidate_wapes or [])],
        },
        dedupe_key=f"forecast_no_model_beats_baseline:{task}:{today_str}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_failed_logins_spike_alert(
    session: Session,
    *,
    ip_address: str,
    failed_count: int,
    window_minutes: int = 10,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when an IP address triggers >= 10 failed logins in 10 minutes (SEC-LOG-03)."""
    return create_alert(
        session=session,
        severity="warning",
        kind=KIND_FAILED_LOGINS_SPIKE,
        message=(
            f"Potential brute-force attack detected: {failed_count} failed logins from IP "
            f"'{ip_address}' within {window_minutes} minutes"
        ),
        payload={
            "ip_address": ip_address,
            "failed_count": failed_count,
            "window_minutes": window_minutes,
        },
        dedupe_key=f"failed_logins_spike:{ip_address}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )


def emit_stockout_spike_alert(
    session: Session,
    *,
    reorder_now_count: int,
    total_skus: int,
    as_of_date: datetime.date | None = None,
    dispatch_webhook: bool = True,
    webhook_url: str | None = None,
) -> tuple[Alert, bool]:
    """Emit an alert when the number of SKUs requiring immediate reorder spikes abnormally."""
    pct = (reorder_now_count / total_skus) if total_skus > 0 else 0.0
    today_str = (as_of_date or utc_now().date()).isoformat()
    return create_alert(
        session=session,
        severity="warning",
        kind=KIND_STOCKOUT_SPIKE,
        message=(
            f"Inventory stockout alert: {reorder_now_count} SKUs ({pct:.1%}) require immediate "
            "reordering (ROP breached)"
        ),
        payload={
            "reorder_now_count": reorder_now_count,
            "total_skus": total_skus,
            "reorder_percentage": round(pct, 4),
        },
        dedupe_key=f"stockout_spike:{today_str}",
        dispatch_webhook=dispatch_webhook,
        webhook_url=webhook_url,
    )
