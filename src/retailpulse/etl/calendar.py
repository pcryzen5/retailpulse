"""Calendar dimension, trading day derivation, and closure window calculations."""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Sequence
from typing import Any

import holidays
import numpy as np
import pandas as pd
import sqlalchemy as sa
from sqlalchemy.orm import Session

from retailpulse.db.models.core import DimDate, DimStore
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Default date range per DATABASE.md §6 and TRD §4.4
DEFAULT_START_DATE = datetime.date(2009, 1, 1)
DEFAULT_END_DATE = datetime.date(2013, 12, 31)

# In Online Retail II, Saturday (ISO weekday 6) has no whole-catalog sales over 52 weeks
DEFAULT_NON_TRADING_WEEKDAYS = (6,)

# Lookback trading days for partial last day determination per TRD §4.4
PARTIAL_DAY_LOOKBACK_TRADING_DAYS = 28
PARTIAL_DAY_PERCENTILE_THRESHOLD = 10.0


def default_christmas_closure_windows(
    start_year: int, end_year: int
) -> list[tuple[datetime.date, datetime.date]]:
    """Generate annual Christmas/New Year closure windows (Dec 24 through Jan 3).

    Per TRD §4.4, the historical data shows recurring 11-12 day closures across Christmas
    and New Year (last trading day Dec 23, reopening Jan 4).
    """
    windows: list[tuple[datetime.date, datetime.date]] = []
    for year in range(start_year, end_year + 1):
        w_start = datetime.date(year, 12, 24)
        w_end = datetime.date(year + 1, 1, 3)
        windows.append((w_start, w_end))
    return windows


def is_in_closure_window(
    d: datetime.date,
    closure_windows: Iterable[tuple[datetime.date, datetime.date]],
) -> bool:
    """Return True if date d falls inside any closure window (inclusive)."""
    return any(start <= d <= end for start, end in closure_windows)


def days_to_christmas(d: datetime.date) -> int:
    """Signed difference in days to the nearest Christmas (Dec 25), clipped to [-30, 30].

    Per TRD §5.1, this feature gives models visibility into the end-of-year peak and closure.
    """
    candidates = [
        datetime.date(d.year - 1, 12, 25),
        datetime.date(d.year, 12, 25),
        datetime.date(d.year + 1, 12, 25),
    ]
    diffs = [(d - c).days for c in candidates]
    closest_diff = min(diffs, key=abs)
    return max(-30, min(30, closest_diff))


def build_calendar_records(
    start_date: datetime.date = DEFAULT_START_DATE,
    end_date: datetime.date = DEFAULT_END_DATE,
    country: str = "GB",
    non_trading_weekdays: Sequence[int] = DEFAULT_NON_TRADING_WEEKDAYS,
    closure_windows: list[tuple[datetime.date, datetime.date]] | None = None,
) -> list[dict[str, Any]]:
    """Build calendar attribute dictionaries for core.dim_date for a given date range.

    Args:
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        country: Country code for public holidays (defaults to 'GB').
        non_trading_weekdays: Weekday numbers (1=Monday, 7=Sunday) considered non-trading.
        closure_windows: Optional explicit list of (start, end) closure windows.
            If None, default annual Christmas-New Year closure windows are applied.

    Returns:
        List of dictionaries corresponding to DimDate attributes.
    """
    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) cannot be after end_date ({end_date})")

    years = list(range(start_date.year, end_date.year + 2))
    gb_holidays = holidays.country_holidays(country, years=years)

    if closure_windows is None:
        closure_windows = default_christmas_closure_windows(start_date.year - 1, end_date.year)

    records: list[dict[str, Any]] = []
    current = start_date
    one_day = datetime.timedelta(days=1)

    while current <= end_date:
        _iso_year, iso_week, iso_day_of_week = current.isocalendar()
        quarter = (current.month - 1) // 3 + 1
        is_weekend = iso_day_of_week in (6, 7)
        is_holiday = current in gb_holidays
        holiday_name = gb_holidays.get(current)

        # Trading day determination:
        # Non-trading if (a) non-trading weekday (Saturday), (b) public holiday, (c) closure window
        in_closure = is_in_closure_window(current, closure_windows)
        is_trading = not (iso_day_of_week in non_trading_weekdays or is_holiday or in_closure)

        records.append(
            {
                "date_key": current,
                "year": current.year,
                "quarter": quarter,
                "month": current.month,
                "iso_week": iso_week,
                "day_of_week": iso_day_of_week,
                "day_of_month": current.day,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "holiday_name": holiday_name,
                "is_trading_day": is_trading,
                "is_complete": True,  # Defaults to True; partial last day updated during ETL
            }
        )
        current += one_day

    return records


def detect_closure_windows(
    sales_dates: Iterable[datetime.date],
    min_gap_days: int = 4,
) -> list[tuple[datetime.date, datetime.date]]:
    """Detect calendar closure windows between active sales dates.

    Per TRD §4.4, a gap of min_gap_days (default 4) or more calendar days between consecutive
    trading dates represents an operational closure window (e.g. Christmas/New Year, Easter).

    Args:
        sales_dates: Iterable of dates on which transactions occurred.
        min_gap_days: Minimum calendar day difference to qualify as a closure window.

    Returns:
        Sorted list of (window_start, window_end) inclusive tuples.
    """
    unique_dates = sorted(set(sales_dates))
    if len(unique_dates) < 2:
        return []

    windows: list[tuple[datetime.date, datetime.date]] = []
    one_day = datetime.timedelta(days=1)

    for i in range(len(unique_dates) - 1):
        d_current = unique_dates[i]
        d_next = unique_dates[i + 1]
        gap_days = (d_next - d_current).days

        if gap_days >= min_gap_days:
            window_start = d_current + one_day
            window_end = d_next - one_day
            windows.append((window_start, window_end))

    return windows


def project_closure_windows(
    closure_windows: list[tuple[datetime.date, datetime.date]],
    future_years: Iterable[int],
) -> list[tuple[datetime.date, datetime.date]]:
    """Project historical closure windows onto future years by matching month and day.

    Per TRD §4.4: "For future dates, the previous year's closure window, matched by month
    and day, is assumed to repeat."

    Args:
        closure_windows: Historical closure windows.
        future_years: Years onto which closures should be projected.

    Returns:
        List of projected (window_start, window_end) tuples.
    """
    projected: list[tuple[datetime.date, datetime.date]] = []
    years_set = set(future_years)

    for w_start, w_end in closure_windows:
        cross_year = w_start.year != w_end.year
        for yr in sorted(years_set):
            try:
                p_start = datetime.date(yr, w_start.month, w_start.day)
            except ValueError:
                # Handle leap day
                p_start = datetime.date(yr, w_start.month, 28)

            end_year = yr + 1 if cross_year else yr
            try:
                p_end = datetime.date(end_year, w_end.month, w_end.day)
            except ValueError:
                p_end = datetime.date(end_year, w_end.month, 28)

            projected.append((p_start, p_end))

    return projected


def is_partial_last_day(
    daily_last_timestamps: dict[datetime.date, datetime.datetime] | pd.Series,
    lookback_trading_days: int = PARTIAL_DAY_LOOKBACK_TRADING_DAYS,
    percentile_threshold: float = PARTIAL_DAY_PERCENTILE_THRESHOLD,
) -> bool:
    """Determine whether the last calendar day in a series of daily last timestamps is partial.

    Per TRD §4.4:
    "The last calendar day of a dataset is partial if its last transaction time is earlier
    than the 10th percentile of the daily last-transaction times over the previous 28 trading days."

    Args:
        daily_last_timestamps: Mapping or Series of date -> maximum timestamp on that date.
        lookback_trading_days: Number of previous trading days to calculate the percentile (28).
        percentile_threshold: Percentile threshold to evaluate against (10.0).

    Returns:
        True if the last day is partial, False if complete.
    """
    if isinstance(daily_last_timestamps, dict):
        series = pd.Series(daily_last_timestamps)
    else:
        series = daily_last_timestamps

    if len(series) < 2:
        return False

    series = series.sort_index()
    last_ts = series.iloc[-1]
    last_seconds = last_ts.hour * 3600 + last_ts.minute * 60 + last_ts.second

    # Take up to lookback_trading_days preceding dates
    prev_series = series.iloc[-(lookback_trading_days + 1) : -1]
    if len(prev_series) == 0:
        return False

    prev_seconds = [ts.hour * 3600 + ts.minute * 60 + ts.second for ts in prev_series]
    threshold_seconds = float(np.percentile(prev_seconds, percentile_threshold))

    return bool(last_seconds < threshold_seconds)


def evaluate_dataset_last_day_completeness(
    timestamps: pd.Series | Iterable[datetime.datetime],
    lookback_trading_days: int = PARTIAL_DAY_LOOKBACK_TRADING_DAYS,
) -> tuple[datetime.date, bool]:
    """Find the last calendar date in a dataset and return (last_date, is_complete).

    Args:
        timestamps: All valid invoice timestamps in the dataset.
        lookback_trading_days: Number of preceding days to check against (default 28).

    Returns:
        tuple (last_date, is_complete) where is_complete = not is_partial.
    """
    series = timestamps if isinstance(timestamps, pd.Series) else pd.Series(list(timestamps))

    clean_ts = pd.to_datetime(series, errors="coerce").dropna()
    if clean_ts.empty:
        raise ValueError("Cannot evaluate last day completeness on empty timestamps")

    dates = clean_ts.dt.date
    daily_last = clean_ts.groupby(dates).max().sort_index()

    last_date = daily_last.index[-1]
    is_partial = is_partial_last_day(
        daily_last,
        lookback_trading_days=lookback_trading_days,
    )

    return last_date, not is_partial


def ensure_dim_date(
    session: Session,
    start_date: datetime.date = DEFAULT_START_DATE,
    end_date: datetime.date = DEFAULT_END_DATE,
    country: str = "GB",
    closure_windows: list[tuple[datetime.date, datetime.date]] | None = None,
) -> int:
    """Ensure core.dim_date is populated for the specified date range.

    Idempotently queries existing dates and inserts any missing records.

    Args:
        session: SQLAlchemy session.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        country: Country for public holidays (default 'GB').
        closure_windows: Optional explicit closure windows.

    Returns:
        Number of date rows inserted.
    """
    existing_stmt = sa.select(DimDate.date_key).where(
        DimDate.date_key.between(start_date, end_date)
    )
    existing_dates = set(session.execute(existing_stmt).scalars().all())

    all_records = build_calendar_records(
        start_date=start_date,
        end_date=end_date,
        country=country,
        closure_windows=closure_windows,
    )

    missing_records = [r for r in all_records if r["date_key"] not in existing_dates]
    if not missing_records:
        logger.debug(
            "dim_date up to date",
            start_date=str(start_date),
            end_date=str(end_date),
            existing=len(existing_dates),
        )
        return 0

    # Batch insert missing records
    session.execute(sa.insert(DimDate), missing_records)
    session.flush()

    logger.info(
        "dim_date populated",
        start_date=str(start_date),
        end_date=str(end_date),
        inserted=len(missing_records),
    )
    return len(missing_records)


def ensure_dim_store(
    session: Session,
    store_code: str = "ONLINE",
    store_name: str = "Online store",
    country: str = "GB",
) -> DimStore:
    """Ensure core.dim_store contains the specified store (default 'ONLINE').

    Args:
        session: SQLAlchemy session.
        store_code: Unique store code (e.g. 'ONLINE').
        store_name: Human-readable store name.
        country: Country code.

    Returns:
        DimStore instance.
    """
    stmt = sa.select(DimStore).where(DimStore.store_code == store_code)
    store = session.execute(stmt).scalar_one_or_none()

    if store is None:
        bind = session.get_bind()
        if bind is not None and bind.dialect.name == "sqlite":
            store = DimStore(
                store_id=1, store_code=store_code, store_name=store_name, country=country
            )
        else:
            store = DimStore(store_code=store_code, store_name=store_name, country=country)
        session.add(store)
        session.flush()
        logger.info("dim_store seeded", store_code=store_code)

    return store


def mark_dim_date_completeness(
    session: Session,
    date_key: datetime.date,
    is_complete: bool,
) -> None:
    """Update core.dim_date.is_complete for a specific date.

    Args:
        session: SQLAlchemy session.
        date_key: Target calendar date.
        is_complete: Boolean completeness flag (False for partial last day).
    """
    stmt = sa.update(DimDate).where(DimDate.date_key == date_key).values(is_complete=is_complete)
    session.execute(stmt)
    session.flush()
    logger.info("marked dim_date completeness", date_key=str(date_key), is_complete=is_complete)
