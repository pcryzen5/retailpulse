"""Rate limiting configuration and state using slowapi.

Implements API §1.1 Conventions and SECURITY §7 (SEC-WEB-03):
- Login: 5/min per IP (plus account lockout).
- Uploads: 10/hour per user.
- Triggers and simulate: 20/hour per user.
- Everything else: 120/min per user.
- 429 responses include Retry-After header and RFC 9457 Problem Details.
"""

from __future__ import annotations

import jwt
from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from retailpulse.logging import get_logger

logger = get_logger("retailpulse.api.limiter")


def get_user_or_ip_identifier(request: Request) -> str:
    """Extract rate limiting key: user ID if authenticated, otherwise remote client IP."""
    # 1. From request state if user was already resolved
    user = getattr(request.state, "user", None)
    if user is not None and hasattr(user, "user_id"):
        return f"user:{user.user_id}"

    # 2. Inspect Authorization Bearer token without requiring signature check
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        try:
            payload = jwt.decode(token, options={"verify_signature": False})
            sub = payload.get("sub")
            if sub:
                return f"user:{sub}"
        except Exception:  # noqa: S110
            pass

    # 3. Fallback to client remote IP address
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(
    key_func=get_user_or_ip_identifier,
    default_limits=["120/minute"],
    headers_enabled=False,
)
