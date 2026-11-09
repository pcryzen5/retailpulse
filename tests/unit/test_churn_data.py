"""Unit tests for churn data, cutoffs, labels, features, and purged splits (P3-03).

Verifies TRD §5.3, PRD §7.4 FR-CH-01, FR-CH-02 & TESTING:
- Monthly cutoff generation and validation.
- Purged temporal split: zero leakage across train, val, test split boundaries.
- Strict feature leakage guard: transactions > cutoff never leak into features <= cutoff.
- Churn label outcome window boundary conditions (T, T + 90].
- Feature calculation exactness: recency, tenure, gaps, overdue ratio, returns, trend.
- Database integration with SQLite in-memory session.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from retailpulse.db.models.core import (
    DimCustomer,
    DimDate,
    DimProduct,
    DimStore,
    FactSalesLine,
)
from retailpulse.db.models.ops import IngestionBatch
from retailpulse.features.churn_features import (
    CHURN_FEATURE_COLUMNS,
    build_churn_features,
    build_churn_features_from_db,
)
from retailpulse.ml.churn.labels import (
    build_churn_dataset,
    compute_churn_labels,
    compute_churn_labels_from_db,
    create_purged_temporal_split,
    generate_monthly_cutoffs,
)


def test_generate_monthly_cutoffs() -> None:
    """Verify monthly cutoff generation starting after 6 months through max_date - 90d."""
    min_date = datetime.date(2009, 12, 1)
    max_date = datetime.date(2011, 12, 9)

    cutoffs = generate_monthly_cutoffs(
        min_date, max_date, min_history_days=180, churn_window_days=90
    )

    # First cutoff after 180 days from 2009-12-01 on 1st of month: 2010-06-01
    assert cutoffs[0] == datetime.date(2010, 6, 1)

    # Last cutoff: 2011-09-01 + 90d = 2011-11-30 <= 2011-12-09
    assert cutoffs[-1] == datetime.date(2011, 9, 1)

    # All cutoffs must be 1st of month and strictly chronologically ascending
    for c in cutoffs:
        assert c.day == 1
        assert c + datetime.timedelta(days=90) <= max_date

    assert cutoffs == sorted(cutoffs)
    assert len(cutoffs) == 16

    # Invalid range should raise ValueError
    with pytest.raises(ValueError, match="must be strictly earlier"):
        generate_monthly_cutoffs(datetime.date(2011, 1, 1), datetime.date(2010, 1, 1))


def test_purged_temporal_split_zero_leakage() -> None:
    """Verify train/val/test split enforces purge gaps so label windows never leak."""
    min_date = datetime.date(2009, 12, 1)
    max_date = datetime.date(2011, 12, 9)
    cutoffs = generate_monthly_cutoffs(
        min_date, max_date, min_history_days=180, churn_window_days=90
    )

    train_cutoffs, val_cutoffs, test_cutoffs = create_purged_temporal_split(
        cutoffs,
        churn_window_days=90,
        val_size_cutoffs=2,
        test_size_cutoffs=1,
    )

    assert len(test_cutoffs) == 1
    assert len(val_cutoffs) == 2
    assert len(train_cutoffs) >= 1

    test_first = test_cutoffs[0]
    val_first = val_cutoffs[0]
    val_last = val_cutoffs[-1]
    train_last = train_cutoffs[-1]

    # PURGE GAP 1: Train labels must end before or at validation feature start
    assert train_last + datetime.timedelta(days=90) <= val_first
    # PURGE GAP 2: Validation labels must end before or at test feature start
    assert val_last + datetime.timedelta(days=90) <= test_first

    # Set disjointness across splits
    assert set(train_cutoffs).isdisjoint(set(val_cutoffs))
    assert set(val_cutoffs).isdisjoint(set(test_cutoffs))
    assert set(train_cutoffs).isdisjoint(set(test_cutoffs))

    # Error when cutoffs are too few
    with pytest.raises(ValueError, match="Insufficient cutoffs"):
        create_purged_temporal_split(cutoffs[:3], val_size_cutoffs=2, test_size_cutoffs=1)


def test_temporal_leakage_features_strictly_lte_cutoff() -> None:
    """Verify that transactions after cutoff date T NEVER leak into feature values at T."""
    t_cutoff = datetime.date(2021, 6, 1)

    # Base transactions before and up to cutoff
    base_rows = [
        # Customer 1: purchases on T - 30 and on T
        {
            "customer_id": 1,
            "invoice_date": datetime.date(2021, 5, 2),
            "invoice_no": "INV-1A",
            "quantity": 2,
            "unit_price": 10.0,
            "sku": "S1",
            "country": "UK",
        },
        {
            "customer_id": 1,
            "invoice_date": datetime.date(2021, 6, 1),
            "invoice_no": "INV-1B",
            "quantity": 3,
            "unit_price": 10.0,
            "sku": "S2",
            "country": "UK",
        },
        # Customer 2: purchase on T - 60
        {
            "customer_id": 2,
            "invoice_date": datetime.date(2021, 4, 2),
            "invoice_no": "INV-2A",
            "quantity": 1,
            "unit_price": 50.0,
            "sku": "S3",
            "country": "UK",
        },
    ]
    df_base = pd.DataFrame(base_rows)
    feat_base = build_churn_features(df_base, t_cutoff)

    # Future transactions AFTER cutoff T
    future_rows = [
        # Customer 1 future huge order
        {
            "customer_id": 1,
            "invoice_date": datetime.date(2021, 6, 2),
            "invoice_no": "INV-1C",
            "quantity": 100,
            "unit_price": 50.0,
            "sku": "S99",
            "country": "UK",
        },
        # Customer 3: brand new customer only active AFTER T
        {
            "customer_id": 3,
            "invoice_date": datetime.date(2021, 6, 5),
            "invoice_no": "INV-3A",
            "quantity": 10,
            "unit_price": 20.0,
            "sku": "S1",
            "country": "UK",
        },
    ]
    df_with_future = pd.concat([df_base, pd.DataFrame(future_rows)], ignore_index=True)
    feat_with_future = build_churn_features(df_with_future, t_cutoff)

    # Assert 1: Customer 3 must NOT exist in features at cutoff T
    assert 3 not in feat_with_future["customer_id"].to_numpy()

    # Assert 2: Features for Customer 1 and 2 must match exactly with or without future data
    pd.testing.assert_frame_equal(feat_base, feat_with_future)


def test_churn_label_window_boundary_conditions() -> None:
    """Verify churn label outcome window strictly covers (T, T + 90 days]."""
    t_cutoff = datetime.date(2021, 6, 1)
    w_days = 90
    # Outcome window: (2021-06-01, 2021-08-30]

    rows = [
        # Customer 101: purchase on T (2021-06-01), no future purchase.
        # Purchase AT cutoff is historical, NOT future -> should be labeled churn = 1.
        {
            "customer_id": 101,
            "invoice_date": datetime.date(2021, 6, 1),
            "invoice_no": "I101-A",
            "quantity": 1,
            "unit_price": 10.0,
        },
        # Customer 102: active prior to T, buys at T + 1 day (2021-06-02) -> churn = 0
        {
            "customer_id": 102,
            "invoice_date": datetime.date(2021, 5, 1),
            "invoice_no": "I102-A",
            "quantity": 1,
            "unit_price": 10.0,
        },
        {
            "customer_id": 102,
            "invoice_date": datetime.date(2021, 6, 2),
            "invoice_no": "I102-B",
            "quantity": 1,
            "unit_price": 10.0,
        },
        # Customer 103: active prior to T, buys at exactly T + 90 days (2021-08-30) -> churn = 0
        {
            "customer_id": 103,
            "invoice_date": datetime.date(2021, 5, 1),
            "invoice_no": "I103-A",
            "quantity": 1,
            "unit_price": 10.0,
        },
        {
            "customer_id": 103,
            "invoice_date": datetime.date(2021, 8, 30),
            "invoice_no": "I103-B",
            "quantity": 1,
            "unit_price": 10.0,
        },
        # Customer 104: active prior to T, buys at T + 91 days (2021-08-31) -> churn = 1
        {
            "customer_id": 104,
            "invoice_date": datetime.date(2021, 5, 1),
            "invoice_no": "I104-A",
            "quantity": 1,
            "unit_price": 10.0,
        },
        {
            "customer_id": 104,
            "invoice_date": datetime.date(2021, 8, 31),
            "invoice_no": "I104-B",
            "quantity": 1,
            "unit_price": 10.0,
        },
        # Customer 105: last purchase > 365 days before T -> INACTIVE -> not in active population
        {
            "customer_id": 105,
            "invoice_date": datetime.date(2020, 5, 1),
            "invoice_no": "I105-A",
            "quantity": 1,
            "unit_price": 10.0,
        },
    ]
    df = pd.DataFrame(rows)
    labels_df = compute_churn_labels(
        df, t_cutoff, churn_window_days=w_days, active_lookback_days=365
    )

    label_map = dict(zip(labels_df["customer_id"], labels_df["churn_label"], strict=True))
    days_map = dict(zip(labels_df["customer_id"], labels_df["days_to_next_purchase"], strict=True))

    # Inactive customer 105 must not be present
    assert 105 not in label_map

    # Customer 101: bought on T, 0 purchases in (T, T+90] -> churned
    assert label_map[101] == 1
    assert np.isnan(days_map[101])

    # Customer 102: bought on T + 1 day -> retained
    assert label_map[102] == 0
    assert days_map[102] == 1.0

    # Customer 103: bought on T + 90 days -> retained
    assert label_map[103] == 0
    assert days_map[103] == 90.0

    # Customer 104: bought on T + 91 days -> churned (outside window)
    assert label_map[104] == 1
    assert np.isnan(days_map[104])


def test_churn_features_calculation_fidelity() -> None:
    """Verify fidelity of behavioral features: gaps, overdue ratio, returns, trend."""
    t_cutoff = datetime.date(2021, 6, 1)

    # Customer 1: 3 purchases + 1 return
    # Purchase 1: T - 100 days (2021-02-21), order INV-A, qty 5, price 10, SKU-1
    # Purchase 2: T - 50 days (2021-04-12), order INV-B, qty 2, price 10, SKU-2
    # Purchase 3: T - 10 days (2021-05-22), order INV-C, qty 3, price 10, SKU-1
    # Return:     T - 5 days (2021-05-27), order INV-CR, qty -1, price 10, SKU-1
    d1 = t_cutoff - datetime.timedelta(days=100)
    d2 = t_cutoff - datetime.timedelta(days=50)
    d3 = t_cutoff - datetime.timedelta(days=10)
    d_ret = t_cutoff - datetime.timedelta(days=5)

    # Customer 2: single purchase (1 order) at T - 40 days
    d_single = t_cutoff - datetime.timedelta(days=40)

    rows = [
        {
            "customer_id": 1,
            "invoice_date": d1,
            "invoice_no": "INV-A",
            "quantity": 5,
            "unit_price": 10.0,
            "sku": "SKU-1",
            "country": "UK",
        },
        {
            "customer_id": 1,
            "invoice_date": d2,
            "invoice_no": "INV-B",
            "quantity": 2,
            "unit_price": 10.0,
            "sku": "SKU-2",
            "country": "UK",
        },
        {
            "customer_id": 1,
            "invoice_date": d3,
            "invoice_no": "INV-C",
            "quantity": 3,
            "unit_price": 10.0,
            "sku": "SKU-1",
            "country": "UK",
        },
        {
            "customer_id": 1,
            "invoice_date": d_ret,
            "invoice_no": "INV-CR",
            "quantity": -1,
            "unit_price": 10.0,
            "sku": "SKU-1",
            "country": "UK",
        },
        {
            "customer_id": 2,
            "invoice_date": d_single,
            "invoice_no": "INV-S",
            "quantity": 4,
            "unit_price": 25.0,
            "sku": "SKU-3",
            "country": "France",
        },
    ]
    df = pd.DataFrame(rows)
    feat_df = build_churn_features(df, t_cutoff)

    c1 = feat_df[feat_df["customer_id"] == 1].iloc[0]
    c2 = feat_df[feat_df["customer_id"] == 2].iloc[0]

    # Customer 1 verification
    assert c1["recency_days"] == 10.0
    assert c1["tenure_days"] == 100.0
    assert c1["orders_30d"] == 1  # only INV-C is in (T-30, T]
    assert c1["orders_90d"] == 2  # INV-B and INV-C in (T-90, T]
    assert c1["orders_180d"] == 3
    assert c1["orders_365d"] == 3
    assert c1["orders_prior_90d"] == 1  # INV-A in (T-180, T-90]
    # Trend formula: (orders_90 - orders_prior_90) / (orders_90 + orders_prior_90)
    assert pytest.approx(c1["order_count_trend"], abs=1e-3) == 1.0 / 3.0
    # Net revenue 365d: 50 + 20 + 30 - 10 = 90.0
    assert c1["net_revenue_365d"] == 90.0
    # Avg order value: 90 / 3 = 30.0
    assert c1["avg_order_value"] == 30.0
    # Distinct SKUs: SKU-1, SKU-2 -> 2
    assert c1["distinct_skus"] == 2
    # Return rate: 1 / (10 + 1) = 1 / 11
    assert pytest.approx(c1["return_rate"], abs=1e-3) == 1.0 / 11.0
    # Gaps: (d2 - d1) = 50 days, (d3 - d2) = 40 days -> mean gap = 45.0
    assert pytest.approx(c1["gap_mean"], abs=1e-3) == 45.0
    # Overdue ratio: recency / gap_mean = 10 / 45 = 0.222
    assert pytest.approx(c1["overdue_ratio"], abs=1e-3) == 10.0 / 45.0
    assert c1["has_single_order"] == 0

    # Customer 2 verification (Single order customer)
    assert c2["orders_365d"] == 1
    assert c2["has_single_order"] == 1
    assert np.isnan(c2["gap_mean"])
    assert np.isnan(c2["gap_std"])
    assert np.isnan(c2["overdue_ratio"])
    assert c2["country"] == "France"
    assert c2["month_of_cutoff"] == 6


def test_build_churn_dataset_multi_cutoff() -> None:
    """Verify build_churn_dataset concatenates multiple cutoffs with labels."""
    c1 = datetime.date(2021, 5, 1)
    c2 = datetime.date(2021, 6, 1)

    rows = [
        {
            "customer_id": 1,
            "invoice_date": datetime.date(2021, 4, 15),
            "invoice_no": "I1",
            "quantity": 1,
            "unit_price": 20.0,
            "sku": "S1",
            "country": "UK",
        },
        {
            "customer_id": 1,
            "invoice_date": datetime.date(2021, 5, 15),
            "invoice_no": "I2",
            "quantity": 1,
            "unit_price": 20.0,
            "sku": "S1",
            "country": "UK",
        },
        {
            "customer_id": 2,
            "invoice_date": datetime.date(2021, 4, 10),
            "invoice_no": "I3",
            "quantity": 1,
            "unit_price": 30.0,
            "sku": "S2",
            "country": "UK",
        },
    ]
    df = pd.DataFrame(rows)
    dataset = build_churn_dataset(df, [c1, c2], churn_window_days=60, lookback_days=180)

    assert "churn_label" in dataset.columns
    assert "cutoff_date" in dataset.columns
    for col in CHURN_FEATURE_COLUMNS:
        assert col in dataset.columns

    # Cutoff 1 has customers 1 and 2
    # Cutoff 2 has customers 1 and 2
    assert len(dataset) == 4
    assert set(dataset["cutoff_date"].unique()) == {c1, c2}


def test_db_integration_churn_features_and_labels() -> None:
    """Verify compute_churn_labels_from_db and build_churn_features_from_db against SQLite."""
    engine = sa.create_engine("sqlite:///:memory:")
    # Translate schemas for SQLite
    engine = engine.execution_options(
        schema_translate_map={"core": None, "ops": None, "ml": None, "app": None}
    )

    # Create tables
    for tbl in [
        DimStore.__table__,
        DimProduct.__table__,
        DimCustomer.__table__,
        DimDate.__table__,
        IngestionBatch.__table__,
        FactSalesLine.__table__,
    ]:
        tbl.create(engine, checkfirst=True)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()

    batch_id = uuid.uuid4()
    session.add(
        IngestionBatch(
            batch_id=batch_id,
            source_name="test",
            source_file="test.xlsx",
            file_sha256="a" * 64,
            status="succeeded",
        )
    )
    session.add(DimStore(store_id=1, store_code="ONLINE", store_name="Online Store"))
    session.add(DimProduct(product_id=1, sku="SKU-1", description="Test SKU 1", category="General"))
    session.add(
        DimCustomer(
            customer_id=10,
            customer_key="CUST-10",
            country="United Kingdom",
            first_order_date=datetime.date(2021, 1, 1),
        )
    )

    # Customer 10 purchases
    # 1. Historical purchase at 2021-04-01
    # 2. Future purchase at 2021-07-01
    lines = [
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h1",
            invoice_no="INV-10A",
            invoice_ts=datetime.datetime(2021, 4, 1, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2021, 4, 1),
            store_id=1,
            product_id=1,
            customer_id=10,
            quantity=5,
            unit_price=Decimal("12.50"),
            is_return=False,
        ),
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h2",
            invoice_no="INV-10B",
            invoice_ts=datetime.datetime(2021, 7, 1, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2021, 7, 1),
            store_id=1,
            product_id=1,
            customer_id=10,
            quantity=2,
            unit_price=Decimal("12.50"),
            is_return=False,
        ),
    ]
    session.add_all(lines)
    session.commit()

    cutoff = datetime.date(2021, 6, 1)

    # 1. Feature extraction from DB (data <= cutoff)
    features_df = build_churn_features_from_db(session, cutoff)
    assert len(features_df) == 1
    row = features_df.iloc[0]
    assert row["customer_id"] == 10
    assert row["orders_365d"] == 1  # Only INV-10A is <= 2021-06-01
    assert row["net_revenue_365d"] == 62.50  # 5 * 12.50
    assert row["recency_days"] == (cutoff - datetime.date(2021, 4, 1)).days

    # 2. Label computation from DB
    labels_df = compute_churn_labels_from_db(session, cutoff, churn_window_days=90)
    assert len(labels_df) == 1
    l_row = labels_df.iloc[0]
    assert l_row["customer_id"] == 10
    # Customer 10 bought at 2021-07-01 which is in (2021-06-01, 2021-08-30] -> retained (0)
    assert l_row["churn_label"] == 0
    assert l_row["days_to_next_purchase"] == 30.0

    session.close()
