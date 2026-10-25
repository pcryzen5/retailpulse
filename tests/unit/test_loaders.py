"""Unit tests for ingestion loaders and idempotent database loading.

Verifies TRD §4.3 & P1-08:
- Dimension upserts (core.dim_store, core.dim_product, core.dim_customer)
- Date dimension population (core.dim_date)
- Chunked sales lines loading with ON CONFLICT (source_row_hash) DO NOTHING
- Quality findings persistence in ops.data_quality_result
- Rejection handling when batch gate fails
- Acceptance criteria: loading mini_retail.csv twice yields identical database row counts
"""

from __future__ import annotations

import datetime
import hashlib
from collections.abc import Generator
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.db.models.core import DimCustomer, DimDate, DimProduct, DimStore, FactSalesLine
from retailpulse.db.models.ops import DataQualityResult, IngestionBatch
from retailpulse.etl.clean import clean_sales_lines
from retailpulse.etl.quality import QualityFinding
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter
from retailpulse.ingestion.loaders import (
    compute_file_sha256,
    ensure_dates_exist,
    get_or_create_batch,
    insert_sales_lines_chunked,
    load_sales_dataset,
    record_quality_findings,
    upsert_customers,
    upsert_products,
    upsert_stores,
)

TEST_SALT = "test-salt-secret-at-least-32-bytes-long!"


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """In-memory SQLite session with schema translate map for isolated testing."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={"schema_translate_map": {"core": None, "ops": None, "app": None}},
    )
    tables = [
        cast(sa.Table, IngestionBatch.__table__),
        cast(sa.Table, DataQualityResult.__table__),
        cast(sa.Table, DimStore.__table__),
        cast(sa.Table, DimProduct.__table__),
        cast(sa.Table, DimCustomer.__table__),
        cast(sa.Table, DimDate.__table__),
        cast(sa.Table, FactSalesLine.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session


def test_compute_file_sha256(tmp_path: Path) -> None:
    """Verify compute_file_sha256 matches standard hashlib result."""
    sample_file = tmp_path / "sample.csv"
    content = b"Invoice,StockCode,Price\n123,456,7.89\n"
    sample_file.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()
    actual = compute_file_sha256(sample_file)
    assert actual == expected
    assert len(actual) == 64


def test_ensure_dates_exist(db_session: Session) -> None:
    """Verify missing dates are populated into core.dim_date idempotently."""
    d1 = datetime.date(2010, 5, 1)
    d2 = datetime.date(2010, 5, 2)
    dates = {d1, d2}

    # First call inserts both dates
    inserted = ensure_dates_exist(db_session, dates)
    assert inserted == 2

    # Second call detects existing dates and inserts 0
    inserted_again = ensure_dates_exist(db_session, dates)
    assert inserted_again == 0

    count = db_session.scalar(sa.select(sa.func.count()).select_from(DimDate))
    assert count == 2


def test_upsert_stores(db_session: Session) -> None:
    """Verify store dimension upserts and code-to-id mapping."""
    store_map = upsert_stores(db_session, {"ONLINE", "FLAGSHIP"})
    assert "ONLINE" in store_map
    assert "FLAGSHIP" in store_map
    assert store_map["ONLINE"] > 0
    assert store_map["FLAGSHIP"] > 0

    # Idempotent call returns identical IDs without duplicate insertion
    store_map_2 = upsert_stores(db_session, {"ONLINE"})
    assert store_map_2["ONLINE"] == store_map["ONLINE"]

    total_stores = db_session.scalar(sa.select(sa.func.count()).select_from(DimStore))
    assert total_stores == 2


def test_upsert_products(db_session: Session) -> None:
    """Verify product dimension upserts, date expansion, and description updates."""
    # Batch 1: SKUs with initial dates
    df1 = pd.DataFrame(
        [
            {
                "sku": "85123A",
                "description": "WHITE HANGING HEART",
                "is_product": True,
                "invoice_ts": pd.Timestamp("2010-06-01 10:00:00"),
            },
            {
                "sku": "POST",
                "description": "POSTAGE",
                "is_product": False,
                "invoice_ts": pd.Timestamp("2010-06-05 12:00:00"),
            },
        ]
    )
    prod_map, new_count = upsert_products(db_session, df1)
    assert new_count == 2
    assert "85123A" in prod_map
    assert "POST" in prod_map

    p1 = db_session.execute(sa.select(DimProduct).where(DimProduct.sku == "85123A")).scalar_one()
    assert p1.first_sale_date == datetime.date(2010, 6, 1)
    assert p1.last_sale_date == datetime.date(2010, 6, 1)
    assert p1.is_product is True

    # Batch 2: Earlier and later date for same SKU
    df2 = pd.DataFrame(
        [
            {
                "sku": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT",
                "is_product": True,
                "invoice_ts": pd.Timestamp("2010-01-15 09:00:00"),
            },
            {
                "sku": "85123A",
                "description": "WHITE HANGING HEART T-LIGHT",
                "is_product": True,
                "invoice_ts": pd.Timestamp("2010-12-20 18:00:00"),
            },
            {
                "sku": "20727",
                "description": "LUNCH BAG BLACK SKULL",
                "is_product": True,
                "invoice_ts": pd.Timestamp("2010-08-01 11:00:00"),
            },
        ]
    )
    prod_map_2, new_count_2 = upsert_products(db_session, df2)
    assert new_count_2 == 1  # only 20727 is new
    assert prod_map_2["85123A"] == p1.product_id

    # Refresh p1 from session
    db_session.refresh(p1)
    assert p1.first_sale_date == datetime.date(2010, 1, 15)
    assert p1.last_sale_date == datetime.date(2010, 12, 20)


def test_upsert_customers(db_session: Session) -> None:
    """Verify customer dimension upserts, date expansion, and null filtering."""
    df1 = pd.DataFrame(
        [
            {
                "customer_key": "c" * 64,
                "country": "United Kingdom",
                "invoice_ts": pd.Timestamp("2010-03-01 10:00:00"),
            },
            {
                "customer_key": None,  # Guest checkout
                "country": "United Kingdom",
                "invoice_ts": pd.Timestamp("2010-03-02 10:00:00"),
            },
        ]
    )
    cust_map, new_count = upsert_customers(db_session, df1)
    assert new_count == 1
    assert "c" * 64 in cust_map

    c1 = db_session.execute(
        sa.select(DimCustomer).where(DimCustomer.customer_key == "c" * 64)
    ).scalar_one()
    assert c1.first_order_date == datetime.date(2010, 3, 1)
    assert c1.last_order_date == datetime.date(2010, 3, 1)

    # Batch 2 extends dates
    df2 = pd.DataFrame(
        [
            {
                "customer_key": "c" * 64,
                "country": "United Kingdom",
                "invoice_ts": pd.Timestamp("2010-11-01 10:00:00"),
            }
        ]
    )
    _, new_count_2 = upsert_customers(db_session, df2)
    assert new_count_2 == 0
    db_session.refresh(c1)
    assert c1.last_order_date == datetime.date(2010, 11, 1)


def test_insert_sales_lines_chunked_and_on_conflict(db_session: Session) -> None:
    """Verify chunked insert with ON CONFLICT (source_row_hash) DO NOTHING."""
    # Seed prerequisites
    store_map = upsert_stores(db_session, {"ONLINE"})
    sample_df = pd.DataFrame(
        [{"sku": "SKU1", "description": "D1", "invoice_ts": "2010-01-01", "is_product": True}]
    )
    prod_map, _ = upsert_products(db_session, sample_df)
    ensure_dates_exist(db_session, {datetime.date(2010, 1, 1)})
    batch, _ = get_or_create_batch(
        db_session,
        source_name="test",
        source_file="test.csv",
        file_sha256="a" * 64,
    )

    row1 = {
        "batch_id": batch.batch_id,
        "source_row_hash": "h1" + "0" * 62,
        "invoice_no": "500001",
        # invoice_ts is source-local timezone-naive per TRD §4.4
        "invoice_ts": datetime.datetime(2010, 1, 1, 12, 0, 0),  # noqa: DTZ001
        "date_key": datetime.date(2010, 1, 1),
        "store_id": store_map["ONLINE"],
        "product_id": prod_map["SKU1"],
        "customer_id": None,
        "quantity": 5,
        "unit_price": 2.50,
        "is_return": False,
        "is_outlier": False,
        "currency": "GBP",
    }
    row2 = {
        "batch_id": batch.batch_id,
        "source_row_hash": "h2" + "0" * 62,
        "invoice_no": "500002",
        # invoice_ts is source-local timezone-naive per TRD §4.4
        "invoice_ts": datetime.datetime(2010, 1, 1, 13, 0, 0),  # noqa: DTZ001
        "date_key": datetime.date(2010, 1, 1),
        "store_id": store_map["ONLINE"],
        "product_id": prod_map["SKU1"],
        "customer_id": None,
        "quantity": 10,
        "unit_price": 1.20,
        "is_return": False,
        "is_outlier": False,
        "currency": "GBP",
    }

    # 1. Insert two rows
    count = insert_sales_lines_chunked(db_session, [row1, row2], chunk_size=1)
    assert count == 2
    total = db_session.scalar(sa.select(sa.func.count()).select_from(FactSalesLine))
    assert total == 2

    # 2. Insert row1 again alongside row3
    row3 = dict(row1)
    row3["source_row_hash"] = "h3" + "0" * 62
    row3["invoice_no"] = "500003"

    insert_sales_lines_chunked(db_session, [row1, row3], chunk_size=1)
    # Total should now be 3 (row1 skipped due to conflict, row3 inserted)
    total_after = db_session.scalar(sa.select(sa.func.count()).select_from(FactSalesLine))
    assert total_after == 3


def test_record_quality_findings(db_session: Session) -> None:
    """Verify persisting findings into ops.data_quality_result without duplicates."""
    batch, _ = get_or_create_batch(
        db_session,
        source_name="test_dq",
        source_file="dq.csv",
        file_sha256="b" * 64,
    )
    findings = [
        QualityFinding(check_name="null_sku", severity="error", passed=True, affected_rows=0),
        QualityFinding(check_name="outlier_qty", severity="warning", passed=True, affected_rows=2),
    ]

    recorded = record_quality_findings(db_session, batch.batch_id, findings)
    assert recorded == 2

    # Calling again with same findings skips existing
    recorded_again = record_quality_findings(db_session, batch.batch_id, findings)
    assert recorded_again == 0

    dq_rows = (
        db_session.execute(
            sa.select(DataQualityResult).where(DataQualityResult.batch_id == batch.batch_id)
        )
        .scalars()
        .all()
    )
    assert len(dq_rows) == 2


def test_load_sales_dataset_rejected_batch(db_session: Session) -> None:
    """Verify dataset with failed batch gate is marked rejected with 0 rows loaded."""
    df = pd.DataFrame(
        [
            {
                "invoice_no": "500001",
                "invoice_ts": pd.Timestamp("2010-01-01 10:00:00"),
                "sku": "SKU1",
                "description": "Product 1",
                "quantity": 5,
                "unit_price": 2.0,
                "customer_key": None,
                "country": "United Kingdom",
                "store_code": "ONLINE",
                "is_return": False,
                "is_product": True,
                "is_outlier": False,
                "source_row_hash": "h_gate" + "0" * 58,
            }
        ]
    )
    findings = [
        QualityFinding(
            check_name="batch_rejection_gate",
            severity="error",
            passed=False,
            affected_rows=100,
        )
    ]

    class FakeDataset:
        def __init__(self) -> None:
            self.df = df
            self.findings = findings
            self.quarantined_df = None
            self.row_counts = {"raw": 100}

    res = load_sales_dataset(
        db_session,
        FakeDataset(),
        source_name="test_gate",
        source_file="gate.csv",
    )
    assert res.status == "rejected"
    assert res.rows_loaded == 0

    fsl_count = db_session.scalar(sa.select(sa.func.count()).select_from(FactSalesLine))
    assert fsl_count == 0

    batch = db_session.get(IngestionBatch, res.batch_id)
    assert batch is not None
    assert batch.status == "rejected"


def test_idempotent_load_fixture_twice(db_session: Session) -> None:
    """Acceptance criteria: loading mini_retail.csv twice yields identical database row counts."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    if not fixture_path.exists():
        pytest.skip("mini_retail.csv fixture not found")

    raw_df = OnlineRetailIIAdapter.read_csv(fixture_path)
    cleaned = clean_sales_lines(raw_df, salt=TEST_SALT)
    source_file = str(fixture_path)
    source_name = "online_retail_ii"

    # --- FIRST LOAD ---
    res1 = load_sales_dataset(
        db_session,
        cleaned,
        source_name=source_name,
        source_file=source_file,
    )
    assert res1.status == "succeeded"
    assert res1.rows_loaded > 0
    assert res1.rows_skipped == 0

    # Capture row counts after First Load
    fsl_count_1 = db_session.scalar(sa.select(sa.func.count()).select_from(FactSalesLine)) or 0
    prod_count_1 = db_session.scalar(sa.select(sa.func.count()).select_from(DimProduct)) or 0
    cust_count_1 = db_session.scalar(sa.select(sa.func.count()).select_from(DimCustomer)) or 0
    store_count_1 = db_session.scalar(sa.select(sa.func.count()).select_from(DimStore)) or 0
    batch_count_1 = db_session.scalar(sa.select(sa.func.count()).select_from(IngestionBatch)) or 0
    dq_count_1 = db_session.scalar(sa.select(sa.func.count()).select_from(DataQualityResult)) or 0

    assert fsl_count_1 == res1.rows_loaded
    assert prod_count_1 > 0
    assert cust_count_1 > 0
    assert store_count_1 >= 1
    assert batch_count_1 == 1
    assert dq_count_1 > 0

    # --- SECOND LOAD (Identical dataset into the same database session) ---
    res2 = load_sales_dataset(
        db_session,
        cleaned,
        source_name=source_name,
        source_file=source_file,
    )
    assert res2.status == "succeeded"
    # All rows should be detected as duplicates and skipped
    assert res2.rows_loaded == 0
    assert res2.rows_skipped == res1.rows_loaded

    # Capture row counts after Second Load
    fsl_count_2 = db_session.scalar(sa.select(sa.func.count()).select_from(FactSalesLine))
    prod_count_2 = db_session.scalar(sa.select(sa.func.count()).select_from(DimProduct))
    cust_count_2 = db_session.scalar(sa.select(sa.func.count()).select_from(DimCustomer))
    store_count_2 = db_session.scalar(sa.select(sa.func.count()).select_from(DimStore))
    batch_count_2 = db_session.scalar(sa.select(sa.func.count()).select_from(IngestionBatch))
    dq_count_2 = db_session.scalar(sa.select(sa.func.count()).select_from(DataQualityResult))

    # CRUCIAL ACCEPTANCE CRITERIA: ALL ROW COUNTS REMAIN EXACTLY IDENTICAL
    assert fsl_count_2 == fsl_count_1, "FactSalesLine row count changed on duplicate load!"
    assert prod_count_2 == prod_count_1, "DimProduct row count changed on duplicate load!"
    assert cust_count_2 == cust_count_1, "DimCustomer row count changed on duplicate load!"
    assert store_count_2 == store_count_1, "DimStore row count changed on duplicate load!"
    assert batch_count_2 == batch_count_1, "IngestionBatch row count changed on duplicate load!"
    assert dq_count_2 == dq_count_1, "DataQualityResult row count changed on duplicate load!"
