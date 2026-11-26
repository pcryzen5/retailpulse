"""Audit logging service enforcing SEC-LOG-01 and SEC-LOG-02.

Records auditable user and system actions in ops.audit_log:
- login_success, login_failure
- upload
- trigger
- promote
- param_update
- user_create, user_update
- alert_ack
- export
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from sqlalchemy.orm import Session

from retailpulse.db.models.ops import AuditLog
from retailpulse.timeutil import utc_now

SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "password",
        "password_hash",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "customer_key",
        "raw_file",
        "file_content",
    }
)


def sanitize_audit_details(details: dict[str, Any] | None) -> dict[str, Any]:
    """Sanitize audit log details to prevent leaking credentials or secrets (SEC-LOG-02)."""
    if not details:
        return {}

    sanitized: dict[str, Any] = {}
    for k, v in details.items():
        if k.lower() in SENSITIVE_KEYS:
            continue
        if isinstance(v, dict):
            sanitized[k] = sanitize_audit_details(v)
        else:
            sanitized[k] = v
    return sanitized


def log_audit(
    session: Session,
    action: str,
    *,
    user_id: uuid.UUID | None = None,
    resource: str | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
    ts: datetime.datetime | None = None,
) -> AuditLog:
    """Create and append an audit trail record in ops.audit_log.

    Args:
        session: Active SQLAlchemy database session.
        action: Audit action name (SEC-LOG-01).
        user_id: Optional ID of the user performing the action.
        resource: Optional target resource identifier (e.g. 'model:forecast:run_id').
        details: Optional contextual metadata (passwords and tokens will be stripped).
        ip_address: Optional client IP address string.
        ts: Optional timestamp; defaults to utc_now().

    Returns:
        The newly created AuditLog instance.
    """
    clean_details = sanitize_audit_details(details)

    log_entry = AuditLog(
        ts=ts if ts is not None else utc_now(),
        user_id=user_id,
        action=action,
        resource=resource,
        ip_address=ip_address,
        details=clean_details,
    )
    session.add(log_entry)
    return log_entry
