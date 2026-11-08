"""Unit tests for RFM feature engineering, scoring, and segmentation (P3-01)."""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sklearn.preprocessing import StandardScaler
from sqlalchemy.orm import sessionmaker

from retailpulse.features.rfm import (
    CANONICAL_SEGMENTS,
    SEGMENT_AT_RISK,
    SEGMENT_CHAMPIONS,
    SEGMENT_HIBERNATING,
    SEGMENT_NEW_PROMISING,
    assign_rule_based_segment,
    compute_rfm_from_db,
    compute_rfm_from_df,
    rfm_df_to_records,
    transform_rfm_for_clustering,
)


@pytest.fixture
def tiny_sales_df() -> pd.DataFrame:
    """Hand-crafted tiny sales dataset with known ground-truth RFM metrics.

    Reference cutoff: as_of = 2025-12-31.
    Lookback window: 730 days (start = 2024-01-01).

    Customers:
    - 101: 3 purchases (Inv 1, 2, 3), total $1500, latest 2025-12-30 (recency 1).
    - 102: 2 purchases (Inv 4, 5), total $400, latest 2025-12-01 (recency 30).
    - 103: 1 purchase (Inv 6), total $50, latest 2025-12-29 (recency 2).
    - 104: 2 purchases + 1 return, total $400 net, latest 2025-06-15 (recency 199).
    - 105: 1 purchase (Inv 9), total $20, latest 2024-02-01 (recency 699).
    - 106: 1 purchase + 1 bigger return, net -$50 -> clipped to $0, recency 91.
    - 107: 1 return ONLY (Inv 12), no purchases -> EXCLUDED from population.
    - 108: 1 purchase in 2023 (Inv 13), > 730 days ago -> EXCLUDED from population.
    - NaN customer_id -> EXCLUDED.
    """
    rows = [
        # Customer 101: Champions
        {
            "customer_id": 101,
            "invoice_no": "INV-1",
            "date_key": datetime.date(2025, 12, 25),
            "quantity": 5,
            "unit_price": 100.0,
            "is_return": False,
        },
        {
            "customer_id": 101,
            "invoice_no": "INV-2",
            "date_key": datetime.date(2025, 12, 28),
            "quantity": 10,
            "unit_price": 50.0,
            "is_return": False,
        },
        {
            "customer_id": 101,
            "invoice_no": "INV-3",
            "date_key": datetime.date(2025, 12, 30),
            "quantity": 2,
            "unit_price": 250.0,
            "is_return": False,
        },
        # Customer 102: Loyal
        {
            "customer_id": 102,
            "invoice_no": "INV-4",
            "date_key": datetime.date(2025, 11, 1),
            "quantity": 2,
            "unit_price": 100.0,
            "is_return": False,
        },
        {
            "customer_id": 102,
            "invoice_no": "INV-5",
            "date_key": datetime.date(2025, 12, 1),
            "quantity": 2,
            "unit_price": 100.0,
            "is_return": False,
        },
        # Customer 103: New / Promising
        {
            "customer_id": 103,
            "invoice_no": "INV-6",
            "date_key": datetime.date(2025, 12, 29),
            "quantity": 1,
            "unit_price": 50.0,
            "is_return": False,
        },
        # Customer 104: At Risk (purchases and return)
        {
            "customer_id": 104,
            "invoice_no": "INV-7",
            "date_key": datetime.date(2025, 6, 1),
            "quantity": 4,
            "unit_price": 100.0,
            "is_return": False,
        },
        {
            "customer_id": 104,
            "invoice_no": "INV-8",
            "date_key": datetime.date(2025, 6, 15),
            "quantity": 1,
            "unit_price": 100.0,
            "is_return": False,
        },
        {
            "customer_id": 104,
            "invoice_no": "INV-8R",
            "date_key": datetime.date(2025, 6, 20),
            "quantity": -1,
            "unit_price": 100.0,
            "is_return": True,
        },
        # Customer 105: Hibernating
        {
            "customer_id": 105,
            "invoice_no": "INV-9",
            "date_key": datetime.date(2024, 2, 1),
            "quantity": 1,
            "unit_price": 20.0,
            "is_return": False,
        },
        # Customer 106: Net negative revenue -> clipped to 0
        {
            "customer_id": 106,
            "invoice_no": "INV-10",
            "date_key": datetime.date(2025, 10, 1),
            "quantity": 1,
            "unit_price": 50.0,
            "is_return": False,
        },
        {
            "customer_id": 106,
            "invoice_no": "INV-11",
            "date_key": datetime.date(2025, 10, 5),
            "quantity": -2,
            "unit_price": 50.0,
            "is_return": True,
        },
        # Customer 107: Return only (should be excluded)
        {
            "customer_id": 107,
            "invoice_no": "INV-12",
            "date_key": datetime.date(2025, 12, 10),
            "quantity": -1,
            "unit_price": 100.0,
            "is_return": True,
        },
        # Customer 108: Outside window (> 730 days)
        {
            "customer_id": 108,
            "invoice_no": "INV-13",
            "date_key": datetime.date(2023, 1, 1),
            "quantity": 5,
            "unit_price": 100.0,
            "is_return": False,
        },
        # Missing customer ID (should be excluded)
        {
            "customer_id": None,
            "invoice_no": "INV-14",
            "date_key": datetime.date(2025, 12, 25),
            "quantity": 10,
            "unit_price": 100.0,
            "is_return": False,
        },
    ]
    return pd.DataFrame(rows)


def test_hand_checked_tiny_dataset_rfm(tiny_sales_df: pd.DataFrame) -> None:
    """Verify RFM feature calculation against hand-checked mathematical truth."""
    as_of = datetime.date(2025, 12, 31)
    rfm = compute_rfm_from_df(tiny_sales_df, as_of=as_of, rfm_window_days=730)

    # 1. Population check: exactly 6 eligible customers
    assert len(rfm) == 6
    assert set(rfm["customer_id"]) == {101, 102, 103, 104, 105, 106}
    assert 107 not in set(rfm["customer_id"])  # only returns
    assert 108 not in set(rfm["customer_id"])  # older than 730 days

    rfm_map = rfm.set_index("customer_id").to_dict(orient="index")

    # Customer 101: 3 orders, $1500, last order 2025-12-30 (1 day ago)
    c101 = rfm_map[101]
    assert c101["recency_days"] == 1
    assert c101["frequency"] == 3
    assert c101["monetary"] == 1500.0
    assert c101["last_order_date"] == datetime.date(2025, 12, 30)
    assert c101["first_order_date"] == datetime.date(2025, 12, 25)
    assert c101["r_score"] == 5
    assert c101["f_score"] == 5
    assert c101["m_score"] == 5
    assert c101["rfm_score"] == "555"
    assert c101["rule_segment"] == SEGMENT_CHAMPIONS

    # Customer 102: 2 orders, $400, last order 2025-12-01 (30 days ago)
    c102 = rfm_map[102]
    assert c102["recency_days"] == 30
    assert c102["frequency"] == 2
    assert c102["monetary"] == 400.0
    assert c102["last_order_date"] == datetime.date(2025, 12, 1)
    assert c102["first_order_date"] == datetime.date(2025, 11, 1)

    # Customer 103: 1 order, $50, last order 2025-12-29 (2 days ago)
    c103 = rfm_map[103]
    assert c103["recency_days"] == 2
    assert c103["frequency"] == 1
    assert c103["monetary"] == 50.0
    assert c103["r_score"] >= 4  # very recent
    assert c103["f_score"] <= 2  # low frequency
    assert c103["rule_segment"] == SEGMENT_NEW_PROMISING

    # Customer 104: 2 orders, $400 net ($500 - $100 return), last order 2025-06-15 (199 days ago)
    c104 = rfm_map[104]
    assert c104["recency_days"] == 199
    assert c104["frequency"] == 2
    assert c104["monetary"] == 400.0
    assert c104["last_order_date"] == datetime.date(2025, 6, 15)
    assert c104["r_score"] <= 2
    assert c104["rule_segment"] == SEGMENT_AT_RISK

    # Customer 105: 1 order, $20, last order 2024-02-01 (699 days ago)
    c105 = rfm_map[105]
    assert c105["recency_days"] == 699
    assert c105["frequency"] == 1
    assert c105["monetary"] == 20.0
    assert c105["r_score"] == 1
    assert c105["rule_segment"] == SEGMENT_HIBERNATING

    # Customer 106: 1 order ($50) + 1 return (-$100) -> net -$50 clipped to 0.0
    c106 = rfm_map[106]
    assert c106["recency_days"] == 91
    assert c106["frequency"] == 1
    assert c106["monetary"] == 0.0  # Clipped >= 0 per TRD §5.2
    assert c106["m_score"] == 1


def test_rule_based_segment_mapping_exhaustive() -> None:
    """Ensure every valid (R, F, M) quintile tuple maps deterministically without gap."""
    seen_segments = set()
    for r in range(1, 6):
        for f in range(1, 6):
            for m in range(1, 6):
                seg = assign_rule_based_segment(r, f, m)
                assert seg in CANONICAL_SEGMENTS
                seen_segments.add(seg)

    assert seen_segments == set(CANONICAL_SEGMENTS)


def test_clustering_transformation_log1p_and_standard_scaler() -> None:
    """Verify log1p + StandardScaler transformation per TRD §5.2."""
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5],
            "recency_days": [1, 10, 30, 90, 365],
            "frequency": [1, 2, 5, 10, 50],
            "monetary": [10.0, 50.0, 200.0, 1000.0, 5000.0],
        }
    )

    x_scaled, scaler = transform_rfm_for_clustering(df)
    assert isinstance(scaler, StandardScaler)
    assert x_scaled.shape == (5, 3)

    # Verify standard scaling properties: zero mean and unit variance
    np.testing.assert_allclose(x_scaled.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(x_scaled.std(axis=0), 1.0, atol=1e-7)

    # Re-apply with fitted scaler
    x_retransformed, _ = transform_rfm_for_clustering(df, scaler=scaler)
    np.testing.assert_allclose(x_scaled, x_retransformed)


def test_clustering_transformation_missing_columns() -> None:
    """Test error when required columns are absent."""
    df = pd.DataFrame({"recency_days": [1, 2], "monetary": [10.0, 20.0]})
    with pytest.raises(ValueError, match="Missing required RFM feature columns"):
        transform_rfm_for_clustering(df)


def test_column_name_variants_support() -> None:
    """Verify compute_rfm_from_df accepts raw Excel/UCI column names."""
    as_of = datetime.date(2025, 12, 31)
    df = pd.DataFrame(
        [
            {
                "Customer ID": 999,
                "Invoice": "50001",
                "InvoiceDate": "2025-12-20 14:30:00",
                "Quantity": 3,
                "Price": 25.0,
            }
        ]
    )
    rfm = compute_rfm_from_df(df, as_of=as_of, rfm_window_days=730)
    assert len(rfm) == 1
    assert rfm.iloc[0]["customer_id"] == 999
    assert rfm.iloc[0]["frequency"] == 1
    assert rfm.iloc[0]["monetary"] == 75.0
    assert rfm.iloc[0]["recency_days"] == 11


def test_empty_dataframe_returns_empty_schema() -> None:
    """Empty inputs should return an empty DataFrame with the exact target schema."""
    as_of = datetime.date(2025, 12, 31)
    rfm = compute_rfm_from_df(pd.DataFrame(), as_of=as_of)
    assert rfm.empty
    assert "customer_id" in rfm.columns
    assert "recency_days" in rfm.columns
    assert "rfm_score" in rfm.columns


def test_rfm_df_to_records_conversion(tiny_sales_df: pd.DataFrame) -> None:
    """Test conversion to strongly-typed CustomerRFM records."""
    as_of = datetime.date(2025, 12, 31)
    rfm_df = compute_rfm_from_df(tiny_sales_df, as_of=as_of)
    records = rfm_df_to_records(rfm_df)
    assert len(records) == 6
    assert records[0].customer_id == 101
    assert records[0].rfm_score == "555"
    assert records[0].rule_segment == SEGMENT_CHAMPIONS


def test_compute_rfm_from_db_mock() -> None:
    """Verify database integration query using an in-memory SQLite session."""
    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    fact_table = sa.Table(
        "fact_sales_line",
        metadata,
        sa.Column("sales_line_id", sa.Integer, primary_key=True),
        sa.Column("customer_id", sa.Integer, nullable=True),
        sa.Column("invoice_no", sa.String, nullable=False),
        sa.Column("date_key", sa.Date, nullable=False),
        sa.Column("quantity", sa.Integer, nullable=False),
        sa.Column("unit_price", sa.Numeric(12, 4), nullable=False),
        sa.Column("line_amount", sa.Numeric(14, 4), nullable=False),
        sa.Column("is_return", sa.Boolean, default=False),
        schema="core",
    )

    with engine.connect() as conn:
        conn.execute(sa.text("ATTACH DATABASE ':memory:' AS core"))
        metadata.create_all(conn)

        conn.execute(
            fact_table.insert(),
            [
                {
                    "customer_id": 201,
                    "invoice_no": "INV-A",
                    "date_key": datetime.date(2025, 12, 25),
                    "quantity": 2,
                    "unit_price": 50.0,
                    "line_amount": 100.0,
                    "is_return": False,
                },
                {
                    "customer_id": 201,
                    "invoice_no": "INV-B",
                    "date_key": datetime.date(2025, 12, 28),
                    "quantity": 1,
                    "unit_price": 50.0,
                    "line_amount": 50.0,
                    "is_return": False,
                },
            ],
        )
        conn.commit()

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        as_of = datetime.date(2025, 12, 31)
        rfm = compute_rfm_from_db(session, as_of=as_of, rfm_window_days=730)
        assert len(rfm) == 1
        assert rfm.iloc[0]["customer_id"] == 201
        assert rfm.iloc[0]["frequency"] == 2
        assert rfm.iloc[0]["monetary"] == 150.0
        assert rfm.iloc[0]["recency_days"] == 3
