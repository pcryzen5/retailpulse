"""Formatting utilities for dashboard display.

Implements DESIGN §2:
- Locale en-GB: currency (£12,345.50, compact £612k, £1.2M)
- Currency symbol derived from RP_CURRENCY environment variable (default '£')
- Number formatting with comma separators or compact metric suffixes (k, M)
- Percentages formatted to 1 decimal place (12.3%)
- Dates formatted in en-GB standard (09 Dec 2011)
"""

from __future__ import annotations

import os
from datetime import date, datetime


def get_currency_symbol() -> str:
    """Retrieve active currency symbol from environment (defaults to £)."""
    return os.getenv("RP_CURRENCY", "£")


def format_currency(
    value: float | int | None,
    *,
    compact: bool = False,
    decimals: int = 2,
) -> str:
    """Format numeric currency value according to en-GB locale rules."""
    if value is None:
        return "-"

    sym = get_currency_symbol()
    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if compact:
        if abs_val >= 1_000_000:
            compact_val = abs_val / 1_000_000
            # Show 1 decimal place for millions unless whole number
            if compact_val == int(compact_val):
                formatted = f"{int(compact_val)}"
            else:
                formatted = f"{compact_val:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{sym}{formatted}M"
        if abs_val >= 1_000:
            compact_val = abs_val / 1_000
            if compact_val == int(compact_val):
                formatted = f"{int(compact_val)}"
            else:
                formatted = f"{compact_val:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{sym}{formatted}k"

    return f"{sign}{sym}{abs_val:,.{decimals}f}"


def format_number(
    value: float | int | None,
    *,
    compact: bool = False,
    decimals: int = 0,
) -> str:
    """Format integer or float value with thousand separators or compact suffix."""
    if value is None:
        return "-"

    abs_val = abs(value)
    sign = "-" if value < 0 else ""

    if compact:
        if abs_val >= 1_000_000:
            compact_val = abs_val / 1_000_000
            if compact_val == int(compact_val):
                formatted = f"{int(compact_val)}"
            else:
                formatted = f"{compact_val:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{formatted}M"
        if abs_val >= 1_000:
            compact_val = abs_val / 1_000
            if compact_val == int(compact_val):
                formatted = f"{int(compact_val)}"
            else:
                formatted = f"{compact_val:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{formatted}k"

    if decimals > 0:
        return f"{sign}{abs_val:,.{decimals}f}"
    return f"{sign}{round(abs_val):,}"


def format_percent(
    value: float | int | None,
    *,
    is_ratio: bool = False,
    decimals: int = 1,
) -> str:
    """Format decimal percentage.

    If is_ratio is True (e.g. 0.123), it is multiplied by 100 -> 12.3%.
    If is_ratio is False (e.g. 12.3), it is formatted directly -> 12.3%.
    """
    if value is None:
        return "-"

    pct_val = value * 100.0 if is_ratio else float(value)
    return f"{pct_val:.{decimals}f}%"


def format_date(value: str | datetime | date | None, fmt: str = "%d %b %Y") -> str:
    """Format date or ISO date string into en-GB format (e.g. 09 Dec 2011)."""
    if value is None or value == "":
        return "-"

    if isinstance(value, (datetime, date)):
        return value.strftime(fmt)

    if isinstance(value, str):
        try:
            clean_str = value.replace("Z", "+00:00")
            parsed_dt = datetime.fromisoformat(clean_str)
            return parsed_dt.strftime(fmt)
        except ValueError:
            return value

    return str(value)
