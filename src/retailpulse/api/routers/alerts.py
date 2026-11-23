"""Alerts and incident management API router.

Implements API §2.6, §3.12:
- GET /api/v1/alerts: Filtered and paginated alerts (Viewer+).
- POST /api/v1/alerts/{alert_id}/ack: Acknowledge alert, audit-logged (Analyst+ only).
"""

from __future__ import annotations

import datetime
import uuid
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import require_role
from retailpulse.api.pagination import (
    PaginatedResponse,
    PaginationParams,
    apply_sort,
    get_pagination_params,
    parse_sort,
)
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.ops import Alert
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError, NotFoundError
from retailpulse.security.audit import log_audit
from retailpulse.timeutil import utc_now

router = APIRouter(tags=["Alerts"])

VALID_SEVERITIES: frozenset[str] = frozenset({"info", "warning", "critical"})
VALID_STATUSES: frozenset[str] = frozenset({"open", "all", "acknowledged"})

ALERT_SORT_WHITELIST: dict[str, Any] = {
    "created_at": Alert.created_at,
    "severity": Alert.severity,
    "kind": Alert.kind,
}


# --- Schemas ---


class AlertItemResponse(BaseModel):
    """Alert record matching API §3.12."""

    model_config = ConfigDict(extra="forbid")

    alert_id: int
    severity: str
    kind: str
    message: str
    payload: dict[str, Any]
    dedupe_key: str | None = None
    created_at: datetime.datetime
    acknowledged_at: datetime.datetime | None = None
    acknowledged_by: uuid.UUID | None = None


# --- Endpoints ---


@router.get(
    "/alerts",
    response_model=PaginatedResponse[AlertItemResponse],
    operation_id="list_alerts",
    summary="List operational alerts",
    dependencies=[Depends(require_role("viewer"))],
)
def list_alerts(
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    status_filter: Annotated[
        str,
        Query(
            alias="status",
            description="Filter by alert status (open, all, acknowledged). Default: open",
        ),
    ] = "open",
    severity: Annotated[
        str | None,
        Query(description="Filter by severity (info, warning, critical)"),
    ] = None,
    sort: Annotated[
        str,
        Query(description="Sort expression. Default: created_at:desc"),
    ] = "created_at:desc",
) -> PaginatedResponse[AlertItemResponse]:
    """Retrieve operational and ML performance alerts with status and severity filters."""
    status_clean = status_filter.strip().lower()
    if status_clean not in VALID_STATUSES:
        raise DomainValidationError(
            f"Invalid status '{status_filter}'. Allowed statuses: {sorted(VALID_STATUSES)}"
        )

    query: sa.Select[Any] = sa.select(Alert)

    if status_clean == "open":
        query = query.where(Alert.acknowledged_at.is_(None))
    elif status_clean == "acknowledged":
        query = query.where(Alert.acknowledged_at.is_not(None))

    if severity:
        sev_clean = severity.strip().lower()
        if sev_clean not in VALID_SEVERITIES:
            raise DomainValidationError(
                f"Invalid severity '{severity}'. Allowed severities: {sorted(VALID_SEVERITIES)}"
            )
        query = query.where(Alert.severity == sev_clean)

    count_stmt = sa.select(sa.func.count()).select_from(query.subquery())
    total = session.scalar(count_stmt) or 0

    sort_clauses = parse_sort(
        sort,
        allowed_fields=ALERT_SORT_WHITELIST.keys(),
        default="created_at:desc",
    )
    query = apply_sort(query, sort_clauses, ALERT_SORT_WHITELIST)
    offset = (pagination.page - 1) * pagination.page_size
    query = query.offset(offset).limit(pagination.page_size)

    rows = session.execute(query).scalars().all()
    items = [
        AlertItemResponse(
            alert_id=a.alert_id,
            severity=a.severity,
            kind=a.kind,
            message=a.message,
            payload=a.payload,
            dedupe_key=a.dedupe_key,
            created_at=a.created_at,
            acknowledged_at=a.acknowledged_at,
            acknowledged_by=a.acknowledged_by,
        )
        for a in rows
    ]

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.post(
    "/alerts/{alert_id}/ack",
    response_model=AlertItemResponse,
    operation_id="acknowledge_alert",
    summary="Acknowledge an operational alert (Analyst+ only)",
)
def acknowledge_alert(
    alert_id: int,
    current_user: Annotated[AppUser, Depends(require_role("analyst"))],
    session: Annotated[Session, Depends(get_db)],
) -> AlertItemResponse:
    """Acknowledge an alert, recording acknowledgment timestamp, user, and audit log."""
    alert = session.get(Alert, alert_id)
    if alert is None:
        raise NotFoundError(f"Alert {alert_id} not found")

    now = utc_now()
    alert.acknowledged_at = now
    alert.acknowledged_by = current_user.user_id

    # SEC-LOG-01: Audit log alert acknowledgment
    log_audit(
        session=session,
        action="alert_ack",
        user_id=current_user.user_id,
        resource=f"alert:{alert_id}",
        details={
            "severity": alert.severity,
            "kind": alert.kind,
            "acknowledged_at": now.isoformat(),
        },
    )

    session.commit()
    session.refresh(alert)

    return AlertItemResponse(
        alert_id=alert.alert_id,
        severity=alert.severity,
        kind=alert.kind,
        message=alert.message,
        payload=alert.payload,
        dedupe_key=alert.dedupe_key,
        created_at=alert.created_at,
        acknowledged_at=alert.acknowledged_at,
        acknowledged_by=alert.acknowledged_by,
    )
