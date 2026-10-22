"""Unit tests for calendar dimension, trading days, closure detection, and partial-day rules."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from retailpulse.db.models.core import DimDate, DimStore
from retailpulse.etl.calendar import (
    build_calendar_records,
    days_to_christmas,
    detect_closure_windows,
    ensure_dim_date,
    ensure_dim_store,
    evaluate_dataset_last_day_completeness,
    is_in_closure_window,
    is_partial_last_day,
    mark_dim_date_completeness,
    project_closure_windows,
)


def test_build_calendar_records_counts_and_attributes() -> None:
    """Verify build_calendar_records generates exact day count and correct attributes."""
    start = datetime.date(2009, 1, 1)
    end = datetime.date(2013, 12, 31)
    records = build_calendar_records(start_date=start, end_date=end)

    # 2009 (365) + 2010 (365) + 2011 (365) + 2012 (366 leap) + 2013 (365) = 1826 days
    assert len(records) == 1826

    first = records[0]
    assert first["date_key"] == start
    assert first["year"] == 2009
    assert first["quarter"] == 1
    assert first["month"] == 1
    assert first["day_of_month"] == 1
    assert first["day_of_week"] == 4  # Thursday
    assert first["is_weekend"] is False
    assert first["is_holiday"] is True  # New Year's Day
    assert "New Year" in (first["holiday_name"] or "")
    assert first["is_trading_day"] is False  # Holiday is non-trading
    assert first["is_complete"] is True

    last = records[-1]
    assert last["date_key"] == end
    assert last["year"] == 2013
    assert last["quarter"] == 4
    assert last["month"] == 12
    assert last["day_of_month"] == 31


def test_invalid_date_range_raises_value_error() -> None:
    """Verify invalid start/end dates raise ValueError."""
    with pytest.raises(ValueError, match="cannot be after"):
        build_calendar_records(
            start_date=datetime.date(2012, 1, 1),
            end_date=datetime.date(2011, 1, 1),
        )


def test_gb_holidays_identification() -> None:
    """Verify GB public holidays are correctly recognized."""
    # Christmas Day 2010
    records = build_calendar_records(
        start_date=datetime.date(2010, 12, 24),
        end_date=datetime.date(2010, 12, 28),
    )
    rec_by_date = {r["date_key"]: r for r in records}

    xmas = rec_by_date[datetime.date(2010, 12, 25)]
    assert xmas["is_holiday"] is True
    assert "Christmas Day" in (xmas["holiday_name"] or "")
    assert xmas["is_trading_day"] is False

    # Normal trading day (e.g. Wednesday 2010-06-16)
    normal = build_calendar_records(
        start_date=datetime.date(2010, 6, 16),
        end_date=datetime.date(2010, 6, 16),
    )[0]
    assert normal["is_holiday"] is False
    assert normal["holiday_name"] is None
    assert normal["is_trading_day"] is True


def test_trading_day_weekend_rules() -> None:
    """Verify Saturday is non-trading while Sunday trades normally in Online Retail II."""
    # 2010-06-18 (Fri), 2010-06-19 (Sat), 2010-06-20 (Sun), 2010-06-21 (Mon)
    records = build_calendar_records(
        start_date=datetime.date(2010, 6, 18),
        end_date=datetime.date(2010, 6, 21),
    )
    rec_by_date = {r["date_key"]: r for r in records}

    fri = rec_by_date[datetime.date(2010, 6, 18)]
    sat = rec_by_date[datetime.date(2010, 6, 19)]
    sun = rec_by_date[datetime.date(2010, 6, 20)]
    mon = rec_by_date[datetime.date(2010, 6, 21)]

    assert fri["day_of_week"] == 5
    assert fri["is_trading_day"] is True

    assert sat["day_of_week"] == 6
    assert sat["is_weekend"] is True
    assert sat["is_trading_day"] is False  # Saturday non-trading

    assert sun["day_of_week"] == 7
    assert sun["is_weekend"] is True
    assert sun["is_trading_day"] is True  # Sunday trades normally

    assert mon["day_of_week"] == 1
    assert mon["is_trading_day"] is True


def test_closure_detection_and_projection() -> None:
    """Verify closure windows detection (>= 4 day gaps) and future projection."""
    sales_dates = [
        datetime.date(2010, 12, 20),
        datetime.date(2010, 12, 21),
        datetime.date(2010, 12, 22),
        datetime.date(2010, 12, 23),
        # Gap from Dec 24 through Jan 3 (11 days)
        datetime.date(2011, 1, 4),
        datetime.date(2011, 1, 5),
        # Gap of 3 days (e.g. 2011-01-05 to 2011-01-08) should NOT be a closure window
        datetime.date(2011, 1, 8),
    ]

    windows = detect_closure_windows(sales_dates, min_gap_days=4)
    assert len(windows) == 1
    start, end = windows[0]
    assert start == datetime.date(2010, 12, 24)
    assert end == datetime.date(2011, 1, 3)

    # Verify is_in_closure_window helper
    assert is_in_closure_window(datetime.date(2010, 12, 24), windows) is True
    assert is_in_closure_window(datetime.date(2010, 12, 31), windows) is True
    assert is_in_closure_window(datetime.date(2011, 1, 3), windows) is True
    assert is_in_closure_window(datetime.date(2011, 1, 4), windows) is False

    # Test future projection onto 2012 and 2013
    projected = project_closure_windows(windows, future_years=[2012, 2013])
    assert len(projected) == 2
    assert projected[0] == (datetime.date(2012, 12, 24), datetime.date(2013, 1, 3))
    assert projected[1] == (datetime.date(2013, 12, 24), datetime.date(2014, 1, 3))


def test_days_to_christmas() -> None:
    """Verify signed and clipped distance to nearest Christmas (Dec 25)."""
    assert days_to_christmas(datetime.date(2010, 12, 25)) == 0
    assert days_to_christmas(datetime.date(2010, 12, 20)) == -5
    assert days_to_christmas(datetime.date(2010, 12, 28)) == 3
    assert days_to_christmas(datetime.date(2011, 1, 4)) == 10
    # Far dates are clipped to [-30, 30]
    assert days_to_christmas(datetime.date(2010, 10, 1)) == -30
    assert days_to_christmas(datetime.date(2011, 3, 1)) == 30


def test_is_partial_last_day_synthetic() -> None:
    """Verify is_partial_last_day detects truncated final days against 10th percentile."""
    base_date = datetime.date(2011, 11, 1)
    daily_last: dict[datetime.date, datetime.datetime] = {}

    # Create 30 days of trading with closing times between 17:00 and 18:00
    for day_offset in range(30):
        d = base_date + datetime.timedelta(days=day_offset)
        # Alternate closing between 17:15 and 17:45
        minute = 15 if day_offset % 2 == 0 else 45
        daily_last[d] = datetime.datetime(
            d.year, d.month, d.day, 17, minute, 0, tzinfo=datetime.UTC
        )

    # 1. When the final day also closes at 17:30 -> Not partial
    d_final_normal = base_date + datetime.timedelta(days=30)
    daily_last[d_final_normal] = datetime.datetime(
        d_final_normal.year,
        d_final_normal.month,
        d_final_normal.day,
        17,
        30,
        0,
        tzinfo=datetime.UTC,
    )
    assert is_partial_last_day(daily_last, lookback_trading_days=28) is False

    # 2. When the final day closes early at 12:50 -> Partial
    daily_last[d_final_normal] = datetime.datetime(
        d_final_normal.year,
        d_final_normal.month,
        d_final_normal.day,
        12,
        50,
        0,
        tzinfo=datetime.UTC,
    )
    assert is_partial_last_day(daily_last, lookback_trading_days=28) is True


def test_fixture_partial_last_day_reconciliation() -> None:
    """Verify the real fixture 2011-12-09 is evaluated as partial (is_complete = False)."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} does not exist.")

    df = pd.read_csv(fixture_path)
    last_date, is_complete = evaluate_dataset_last_day_completeness(df["InvoiceDate"])

    assert last_date == datetime.date(2011, 12, 9)
    # Per TRD §4.4: 2011-12-09 ends at 12:50 and is partial, so is_complete must be False
    assert is_complete is False


def test_ensure_dim_date_and_dim_store_sqlite() -> None:
    """Verify ensure_dim_date and ensure_dim_store work idempotently with SQLAlchemy."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={"schema_translate_map": {"core": None}},
    )
    DimStore.metadata.create_all(
        engine,
        tables=[cast(sa.Table, DimStore.__table__), cast(sa.Table, DimDate.__table__)],
    )

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        # 1. Test ensure_dim_store creates ONLINE store
        store = ensure_dim_store(session)
        assert store.store_code == "ONLINE"
        assert store.country == "GB"
        session.commit()

        # Idempotent call returns existing store
        store_again = ensure_dim_store(session)
        assert store_again.store_id == store.store_id

        # 2. Test ensure_dim_date inserts missing rows
        start = datetime.date(2010, 1, 1)
        end = datetime.date(2010, 1, 10)
        inserted = ensure_dim_date(session, start_date=start, end_date=end)
        assert inserted == 10
        session.commit()

        # Running again inserts 0 (idempotent)
        inserted_again = ensure_dim_date(session, start_date=start, end_date=end)
        assert inserted_again == 0

        # Verify row count in DB
        count = session.execute(sa.select(sa.func.count(DimDate.date_key))).scalar_one()
        assert count == 10

        # 3. Test mark_dim_date_completeness
        mark_dim_date_completeness(session, end, is_complete=False)
        session.commit()

        is_comp = session.execute(
            sa.select(DimDate.is_complete).where(DimDate.date_key == end)
        ).scalar_one()
        assert is_comp is False
