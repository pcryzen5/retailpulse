"""Time and calendar utilities for RetailPulse system code."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return the current time in UTC with timezone awareness.

    Note:
        This is the sole authorized clock access point in the codebase.
        ML, features, and ETL logic must never call this; they must take
        explicit cutoff/as_of parameters.
    """
    return datetime.now(UTC)
