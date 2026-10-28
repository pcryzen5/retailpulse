"""Unit tests for forecast features, top-N selection, eligibility, and zero-filled series.

Verifies P2-01 & TRD §5.1:
- Top-N selection by trailing-365-day revenue up to as_of
- Eligibility determination: first_sale >= 90 days ago and >= 20 non-zero sales days
- Zero-filled trading-day series generation (sparse fact_daily_sales filled on trading days)
- Lumpy order target clipping (99.5th percentile per series)
- Feature engineering: lags (1, 7, 14, 28), rolling stats (7, 28), days since last sale,
  price ratio, target calendar attributes, horizon day
- Leakage test: features computed at origin use strictly no data after origin
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import cast

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.db.models.core import DimDate, DimProduct, DimStore, FactDailySales
from retailpulse.features.forecast_features import (
    build_direct_features_at_origin,
    build_direct_training_dataset,
    build_zero_filled_series,
    build_zero_filled_series_df,
    clip_training_targets,
    evaluate_product_eligibility,
    evaluate_product_eligibility_df,
    select_top_products,
    select_top_products_df,
)


@pytest.fixture
def memory_db() -> Session:
    """In-memory SQLite session with schema translation map."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={
            "schema_translate_map": {"core": None, "ops": None, "ml": None, "app": None}
        },
    )
    tables = [
        cast(sa.Table, DimStore.__table__),
        cast(sa.Table, DimProduct.__table__),
        cast(sa.Table, DimDate.__table__),
        cast(sa.Table, FactDailySales.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    # Seed single store
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store", country="GB")
    session.add(store)
    session.commit()

    yield session
    session.close()


def test_select_top_products_df() -> None:
    """Verify DataFrame-based top-N selection by trailing-365-day revenue up to as_of."""
    as_of = datetime.date(2011, 12, 8)
    trailing_start = as_of - datetime.timedelta(days=364)

    daily_sales = pd.DataFrame(
        [
            {"product_id": 1, "date_key": as_of - datetime.timedelta(days=10), "revenue": 1000.0},
            {"product_id": 2, "date_key": as_of - datetime.timedelta(days=20), "revenue": 500.0},
            {
                "product_id": 3,
                "date_key": trailing_start - datetime.timedelta(days=5),
                "revenue": 2000.0,
            },
            {"product_id": 2, "date_key": as_of + datetime.timedelta(days=1), "revenue": 5000.0},
        ]
    )

    product_meta = pd.DataFrame(
        [
            {"product_id": 1, "sku": "SKU1", "is_product": True},
            {"product_id": 2, "sku": "SKU2", "is_product": True},
            {"product_id": 3, "sku": "SKU3", "is_product": True},
            {"product_id": 4, "sku": "POST", "is_product": False},
        ]
    )

    top_prods = select_top_products_df(
        daily_sales_df=daily_sales,
        product_meta_df=product_meta,
        as_of=as_of,
        top_n=2,
    )
    assert top_prods == [1, 2]


def test_select_top_products_db(memory_db: Session) -> None:
    """Verify DB-based top-N selection by trailing revenue."""
    as_of = datetime.date(2011, 12, 8)

    p1 = DimProduct(product_id=1, sku="SKU1", is_product=True)
    p2 = DimProduct(product_id=2, sku="SKU2", is_product=True)
    p3 = DimProduct(product_id=3, sku="POST", is_product=False)
    memory_db.add_all([p1, p2, p3])
    memory_db.commit()

    s1 = FactDailySales(
        date_key=as_of - datetime.timedelta(days=5),
        store_id=1,
        product_id=1,
        units_sold=10,
        revenue=Decimal("150.00"),
    )
    s2 = FactDailySales(
        date_key=as_of - datetime.timedelta(days=5),
        store_id=1,
        product_id=2,
        units_sold=20,
        revenue=Decimal("300.00"),
    )
    s3 = FactDailySales(
        date_key=as_of - datetime.timedelta(days=5),
        store_id=1,
        product_id=3,
        units_sold=50,
        revenue=Decimal("1000.00"),
    )
    memory_db.add_all([s1, s2, s3])
    memory_db.commit()

    top_products = select_top_products(memory_db, as_of=as_of, top_n=5)
    assert len(top_products) == 2
    assert top_products[0].product_id == 2
    assert top_products[1].product_id == 1


def test_evaluate_product_eligibility_rules() -> None:
    """Verify eligibility rules: history >= 90 days and non-zero sales days >= 20."""
    as_of = datetime.date(2011, 12, 8)

    records = []
    for i in range(25):
        records.append(
            {
                "product_id": 1,
                "date_key": as_of - datetime.timedelta(days=i * 2 + 1),
                "units_sold": 5,
            }
        )
    for i in range(25):
        records.append(
            {"product_id": 2, "date_key": as_of - datetime.timedelta(days=i + 1), "units_sold": 5}
        )
    for i in range(15):
        records.append(
            {
                "product_id": 3,
                "date_key": as_of - datetime.timedelta(days=i * 10 + 1),
                "units_sold": 5,
            }
        )

    daily_sales = pd.DataFrame(records)
    product_meta = pd.DataFrame(
        [
            {
                "product_id": 1,
                "sku": "SKU1",
                "first_sale_date": as_of - datetime.timedelta(days=120),
                "is_product": True,
            },
            {
                "product_id": 2,
                "sku": "SKU2",
                "first_sale_date": as_of - datetime.timedelta(days=60),
                "is_product": True,
            },
            {
                "product_id": 3,
                "sku": "SKU3",
                "first_sale_date": as_of - datetime.timedelta(days=200),
                "is_product": True,
            },
            {
                "product_id": 4,
                "sku": "POST",
                "first_sale_date": as_of - datetime.timedelta(days=200),
                "is_product": False,
            },
        ]
    )

    eligibility = evaluate_product_eligibility_df(
        daily_sales_df=daily_sales,
        product_meta_df=product_meta,
        as_of=as_of,
    )

    assert eligibility[1].is_eligible is True
    assert eligibility[1].ineligibility_reason is None
    assert eligibility[1].sales_days_count == 25

    assert eligibility[2].is_eligible is False
    assert eligibility[2].ineligibility_reason == "history_too_short"

    assert eligibility[3].is_eligible is False
    assert eligibility[3].ineligibility_reason == "insufficient_sales_days"
    assert eligibility[3].sales_days_count == 15

    assert eligibility[4].is_eligible is False
    assert eligibility[4].ineligibility_reason == "not_product"


def test_build_zero_filled_series_df() -> None:
    """Verify sparse daily sales are zero-filled on all trading days."""
    as_of = datetime.date(2011, 1, 10)
    start_date = datetime.date(2011, 1, 1)

    cur = start_date
    cal_rows = []
    while cur <= as_of:
        is_sat = cur.weekday() == 5
        cal_rows.append({"date_key": cur, "is_trading_day": not is_sat, "is_holiday": False})
        cur += datetime.timedelta(days=1)
    cal_df = pd.DataFrame(cal_rows)

    daily_sales = pd.DataFrame(
        [
            {
                "product_id": 1,
                "date_key": datetime.date(2011, 1, 3),
                "units_sold": 10,
                "avg_unit_price": 2.50,
            },
            {
                "product_id": 1,
                "date_key": datetime.date(2011, 1, 7),
                "units_sold": 15,
                "avg_unit_price": 2.50,
            },
        ]
    )

    meta_df = pd.DataFrame([{"product_id": 1, "sku": "SKU1", "abc_class": "A"}])

    series_df = build_zero_filled_series_df(
        daily_sales_df=daily_sales,
        calendar_df=cal_df,
        product_meta_df=meta_df,
        product_ids=[1],
        start_date=start_date,
        as_of=as_of,
    )

    assert datetime.date(2011, 1, 1) not in series_df["date"].to_numpy()
    assert datetime.date(2011, 1, 8) not in series_df["date"].to_numpy()
    assert len(series_df) == 8

    row_04 = series_df[series_df["date"] == datetime.date(2011, 1, 4)].iloc[0]
    assert row_04["y"] == 0
    assert row_04["price"] == pytest.approx(2.50)
    assert row_04["abc_class"] == "A"

    row_03 = series_df[series_df["date"] == datetime.date(2011, 1, 3)].iloc[0]
    assert row_03["y"] == 10
    assert row_03["price"] == pytest.approx(2.50)


def test_clip_training_targets() -> None:
    """Verify lumpy wholesale spikes are clipped at 99.5th percentile per series."""
    values = [1.0] * 995 + [1000.0] * 5
    df = pd.DataFrame({"series_id": [1] * 1000, "y": values})

    clipped_df = clip_training_targets(df, target_col="y", group_col="series_id", quantile=0.995)
    max_val = clipped_df["y"].max()
    assert max_val < 1000.0


def test_direct_features_at_origin_leakage_safety() -> None:
    """CRITICAL VERIFY TEST: Features computed at origin use strictly no data after origin."""
    base_date = datetime.date(2011, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(100)]
    cal_df = pd.DataFrame(
        [
            {
                "date_key": d,
                "is_trading_day": True,
                "is_holiday": False,
                "day_of_week": d.isocalendar()[2],
                "month": d.month,
                "iso_week": d.isocalendar()[1],
            }
            for d in dates
        ]
    )

    rng = np.random.default_rng(42)
    y_vals = [int(10 + (i % 7) * 3 + rng.integers(0, 5)) for i in range(100)]
    prices = [2.50] * 100

    series_df = pd.DataFrame(
        {
            "series_id": [101] * 100,
            "product_id": [101] * 100,
            "date": dates,
            "y": y_vals,
            "price": prices,
            "is_holiday": [False] * 100,
            "abc_class": ["A"] * 100,
        }
    )

    origin = dates[70]

    features_full = build_direct_features_at_origin(
        history_df=series_df,
        calendar_df=cal_df,
        origin=origin,
        horizon=28,
    )

    truncated_series = series_df[series_df["date"] <= origin].copy()
    features_truncated = build_direct_features_at_origin(
        history_df=truncated_series,
        calendar_df=cal_df,
        origin=origin,
        horizon=28,
    )

    altered_series = series_df.copy()
    altered_series.loc[altered_series["date"] > origin, "y"] = 99999
    altered_series.loc[altered_series["date"] > origin, "price"] = 999.0
    features_altered = build_direct_features_at_origin(
        history_df=altered_series,
        calendar_df=cal_df,
        origin=origin,
        horizon=28,
    )

    feature_cols = [c for c in features_full.columns if c not in ("y_target", "target_date")]

    pd.testing.assert_frame_equal(
        features_full[feature_cols].reset_index(drop=True),
        features_truncated[feature_cols].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        features_full[feature_cols].reset_index(drop=True),
        features_altered[feature_cols].reset_index(drop=True),
    )

    assert features_full["lag_1"].iloc[0] == series_df[series_df["date"] == origin]["y"].iloc[0]
    assert list(features_full["horizon_day"]) == list(range(1, 29))
    assert all(d > origin for d in features_full["target_date"])

    expected_future_y = [
        series_df[series_df["date"] == origin + datetime.timedelta(days=h)]["y"].iloc[0]
        for h in range(1, 29)
    ]
    assert list(features_full["y_target"]) == expected_future_y


def test_build_direct_training_dataset_sampling() -> None:
    """Verify weekly sampling of training origins across a continuous series."""
    base_date = datetime.date(2011, 1, 1)
    dates = [base_date + datetime.timedelta(days=i) for i in range(120)]
    cal_df = pd.DataFrame(
        [
            {
                "date_key": d,
                "is_trading_day": True,
                "is_holiday": False,
                "day_of_week": d.isocalendar()[2],
                "month": d.month,
                "iso_week": d.isocalendar()[1],
            }
            for d in dates
        ]
    )

    series_df = pd.DataFrame(
        {
            "series_id": [1] * 120,
            "product_id": [1] * 120,
            "date": dates,
            "y": [5] * 120,
            "price": [10.0] * 120,
            "is_holiday": [False] * 120,
            "abc_class": ["A"] * 120,
        }
    )

    train_df = build_direct_training_dataset(
        series_df=series_df,
        calendar_df=cal_df,
        horizon=28,
        min_history_days=35,
        sample_step_days=7,
    )

    assert not train_df.empty
    assert (train_df["y_target"] == 5).all()
    origins = sorted(train_df["origin"].unique())
    diffs = [(origins[i + 1] - origins[i]).days for i in range(len(origins) - 1)]
    assert all(diff == 7 for diff in diffs)


def test_evaluate_product_eligibility_db(memory_db: Session) -> None:
    """Verify database-driven product eligibility evaluation."""
    as_of = datetime.date(2011, 12, 8)
    p1 = DimProduct(
        product_id=10,
        sku="SKU10",
        is_product=True,
        first_sale_date=as_of - datetime.timedelta(days=100),
    )
    p2 = DimProduct(
        product_id=20,
        sku="SKU20",
        is_product=True,
        first_sale_date=as_of - datetime.timedelta(days=50),
    )
    memory_db.add_all([p1, p2])
    memory_db.commit()

    sales_rows = [
        FactDailySales(
            date_key=as_of - datetime.timedelta(days=i * 2 + 1),
            store_id=1,
            product_id=10,
            units_sold=5,
            revenue=Decimal("25.00"),
        )
        for i in range(22)
    ]
    memory_db.add_all(sales_rows)
    memory_db.commit()

    eligibility = evaluate_product_eligibility(memory_db, as_of=as_of, product_ids=[10, 20])
    assert eligibility[10].is_eligible is True
    assert eligibility[10].sales_days_count == 22
    assert eligibility[20].is_eligible is False
    assert eligibility[20].ineligibility_reason == "history_too_short"


def test_build_zero_filled_series_db(memory_db: Session) -> None:
    """Verify database-driven zero-filled series generation on trading days."""
    as_of = datetime.date(2011, 1, 5)
    start_date = datetime.date(2011, 1, 1)

    p = DimProduct(product_id=30, sku="SKU30", is_product=True, abc_class="B")
    memory_db.add(p)

    cur = start_date
    while cur <= as_of:
        is_trading = cur.weekday() != 5
        memory_db.add(
            DimDate(
                date_key=cur,
                year=cur.year,
                quarter=1,
                month=cur.month,
                iso_week=cur.isocalendar()[1],
                day_of_week=cur.isocalendar()[2],
                day_of_month=cur.day,
                is_weekend=cur.weekday() in (5, 6),
                is_holiday=False,
                is_trading_day=is_trading,
                is_complete=True,
            )
        )
        cur += datetime.timedelta(days=1)

    memory_db.add(
        FactDailySales(
            date_key=datetime.date(2011, 1, 3),
            store_id=1,
            product_id=30,
            units_sold=12,
            avg_unit_price=Decimal("4.50"),
            revenue=Decimal("54.00"),
        )
    )
    memory_db.commit()

    series = build_zero_filled_series(
        session=memory_db,
        product_ids=[30],
        start_date=start_date,
        as_of=as_of,
        store_id=1,
    )

    assert not series.empty
    assert (series["series_id"] == 30).all()
    assert (series["abc_class"] == "B").all()
    assert len(series) == 4
    jan3_row = series[series["date"] == datetime.date(2011, 1, 3)].iloc[0]
    assert jan3_row["y"] == 12
    assert jan3_row["price"] == pytest.approx(4.50)
