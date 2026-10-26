"""Unit tests for daily sales aggregation and product classification.

Verifies P1-09 & TRD §4.4:
- fact_daily_sales for touched dates
- Exclusion of outlier lines (is_outlier = true)
- Exclusion of incomplete days (dim_date.is_complete = false)
- Exclusion of non-product SKUs (dim_product.is_product = false)
- Product first_sale_date and last_sale_date derivation
- Customer first_order_date and last_order_date derivation
- ABC product classification (trailing-365-day revenue: A <= 80%, B <= 95%, C rest)
- XYZ product classification (weekly demand CV: X < 0.5, Y < 1.0, Z rest)
- Verification criterion: Reconciliation test:
  sum(units_sold) == sum(positive, non-outlier product-line quantity on complete days)
"""

from __future__ import annotations

import datetime
from collections.abc import Generator
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.db.models.core import (
    DimCustomer,
    DimDate,
    DimProduct,
    DimStore,
    FactDailySales,
    FactSalesLine,
)
from retailpulse.db.models.ops import DataQualityResult, IngestionBatch
from retailpulse.etl.aggregate import (
    aggregate_daily_sales,
    classify_abc_xyz,
    compute_customer_order_dates,
    compute_product_first_last_dates,
    run_aggregation_pipeline,
)
from retailpulse.etl.calendar import (
    ensure_dim_date,
    evaluate_dataset_last_day_completeness,
    mark_dim_date_completeness,
)
from retailpulse.etl.clean import clean_sales_lines
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter
from retailpulse.ingestion.loaders import load_sales_dataset

TEST_SALT = "test-salt-secret-at-least-32-bytes-long!"


@pytest.fixture
def db_session() -> Generator[Session, None, None]:
    """In-memory SQLite session with schema translation map for testing."""
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
        cast(sa.Table, FactDailySales.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)
    with session_factory() as session:
        yield session


def _seed_basic_dimensions(session: Session) -> tuple[DimStore, DimProduct, DimProduct, DimProduct]:
    """Seed store and products (two real products, one non-product)."""
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store", country="GB")
    p1 = DimProduct(product_id=1, sku="SKU001", description="Product One", is_product=True)
    p2 = DimProduct(product_id=2, sku="SKU002", description="Product Two", is_product=True)
    p_non = DimProduct(product_id=3, sku="POST", description="Postage fee", is_product=False)
    session.add_all([store, p1, p2, p_non])
    session.flush()
    return store, p1, p2, p_non


def _create_batch(session: Session, source_name: str = "test") -> IngestionBatch:
    """Create a sample ingestion batch record."""
    batch = IngestionBatch(
        source_name=source_name,
        source_file="test.csv",
        file_sha256="a" * 64,
        status="succeeded",
    )
    session.add(batch)
    session.flush()
    return batch


def _ts(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime.datetime:
    """Create timezone-naive timestamp matching source-local invoice_ts convention."""
    return datetime.datetime(year, month, day, hour, minute)  # noqa: DTZ001


def test_aggregate_daily_sales_basic(db_session: Session) -> None:
    """Verify standard aggregation of sales lines into daily sales records."""
    store, p1, _p2, _ = _seed_basic_dimensions(db_session)
    d1 = datetime.date(2010, 6, 1)
    ensure_dim_date(db_session, d1, d1)

    batch = _create_batch(db_session)

    # Product 1: Two sales lines and one return line
    line1 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash01",
        invoice_no="INV100",
        invoice_ts=_ts(2010, 6, 1, 10, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=5,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    line2 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash02",
        invoice_no="INV101",
        invoice_ts=_ts(2010, 6, 1, 11, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=3,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    line3 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash03",
        invoice_no="CINV100",
        invoice_ts=_ts(2010, 6, 1, 12, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=-2,
        unit_price=Decimal("10.00"),
        is_return=True,
        is_outlier=False,
    )

    db_session.add_all([line1, line2, line3])
    db_session.flush()

    count = aggregate_daily_sales(db_session, [d1])
    assert count == 1

    fds = db_session.execute(sa.select(FactDailySales)).scalar_one()
    assert fds.date_key == d1
    assert fds.store_id == store.store_id
    assert fds.product_id == p1.product_id
    assert fds.units_sold == 8  # 5 + 3
    assert fds.units_returned == 2  # |-2|
    assert fds.net_units == 6  # 8 - 2
    assert fds.revenue == Decimal("80.0000")
    assert fds.avg_unit_price == Decimal("10.0000")
    assert fds.n_orders == 2  # INV100, INV101


def test_aggregate_daily_sales_excludes_outliers(db_session: Session) -> None:
    """Verify lines marked is_outlier = true are excluded from fact_daily_sales."""
    store, p1, _, _ = _seed_basic_dimensions(db_session)
    d1 = datetime.date(2010, 6, 1)
    ensure_dim_date(db_session, d1, d1)

    batch = _create_batch(db_session)

    # Normal line
    normal_line = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_norm",
        invoice_no="INV001",
        invoice_ts=_ts(2010, 6, 1, 10, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=10,
        unit_price=Decimal("5.00"),
        is_return=False,
        is_outlier=False,
    )
    # Huge outlier order (e.g. 80,000 units)
    outlier_line = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_out",
        invoice_no="INV002",
        invoice_ts=_ts(2010, 6, 1, 10, 30),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=80000,
        unit_price=Decimal("5.00"),
        is_return=False,
        is_outlier=True,
    )

    db_session.add_all([normal_line, outlier_line])
    db_session.flush()

    aggregate_daily_sales(db_session, [d1])

    fds = db_session.execute(sa.select(FactDailySales)).scalar_one()
    # Outlier must be excluded
    assert fds.units_sold == 10
    assert fds.revenue == Decimal("50.0000")


def test_aggregate_daily_sales_excludes_incomplete_days(db_session: Session) -> None:
    """Verify dates where dim_date.is_complete = false are excluded from daily sales."""
    store, p1, _, _ = _seed_basic_dimensions(db_session)
    d_complete = datetime.date(2010, 6, 1)
    d_partial = datetime.date(2010, 6, 2)
    ensure_dim_date(db_session, d_complete, d_partial)
    mark_dim_date_completeness(db_session, d_partial, is_complete=False)

    batch = _create_batch(db_session)

    line_complete = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_comp",
        invoice_no="INV001",
        invoice_ts=_ts(2010, 6, 1, 10, 0),
        date_key=d_complete,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=5,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    line_partial = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_part",
        invoice_no="INV002",
        invoice_ts=_ts(2010, 6, 2, 10, 0),
        date_key=d_partial,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=5,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )

    db_session.add_all([line_complete, line_partial])
    db_session.flush()

    aggregate_daily_sales(db_session, [d_complete, d_partial])

    records = db_session.execute(sa.select(FactDailySales)).scalars().all()
    assert len(records) == 1
    assert records[0].date_key == d_complete


def test_aggregate_daily_sales_excludes_non_product_skus(db_session: Session) -> None:
    """Verify non-product SKUs (is_product = false) are not aggregated into fact_daily_sales."""
    store, p1, _, p_non = _seed_basic_dimensions(db_session)
    d1 = datetime.date(2010, 6, 1)
    ensure_dim_date(db_session, d1, d1)

    batch = _create_batch(db_session)

    line_prod = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_prod",
        invoice_no="INV001",
        invoice_ts=_ts(2010, 6, 1, 10, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=4,
        unit_price=Decimal("15.00"),
        is_return=False,
        is_outlier=False,
    )
    line_non_prod = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_fee",
        invoice_no="INV001",
        invoice_ts=_ts(2010, 6, 1, 10, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p_non.product_id,
        quantity=1,
        unit_price=Decimal("4.99"),
        is_return=False,
        is_outlier=False,
    )

    db_session.add_all([line_prod, line_non_prod])
    db_session.flush()

    aggregate_daily_sales(db_session, [d1])

    records = db_session.execute(sa.select(FactDailySales)).scalars().all()
    assert len(records) == 1
    assert records[0].product_id == p1.product_id


def test_idempotency_of_daily_sales_aggregation(db_session: Session) -> None:
    """Verify aggregating twice yields identical row counts and values."""
    store, p1, _p2, _ = _seed_basic_dimensions(db_session)
    d1 = datetime.date(2010, 6, 1)
    ensure_dim_date(db_session, d1, d1)

    batch = _create_batch(db_session)

    line1 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="hash_01",
        invoice_no="INV001",
        invoice_ts=_ts(2010, 6, 1, 10, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=5,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    db_session.add(line1)
    db_session.flush()

    # First run
    c1 = aggregate_daily_sales(db_session, [d1])
    assert c1 == 1

    # Second run
    c2 = aggregate_daily_sales(db_session, [d1])
    assert c2 == 1

    count = db_session.scalar(sa.select(sa.func.count()).select_from(FactDailySales))
    assert count == 1


def test_product_first_last_dates_computation(db_session: Session) -> None:
    """Verify first_sale_date and last_sale_date updates on dim_product."""
    store, p1, _, _ = _seed_basic_dimensions(db_session)
    d1 = datetime.date(2010, 3, 10)
    d2 = datetime.date(2010, 7, 20)
    ensure_dim_date(db_session, d1, d2)

    batch = _create_batch(db_session)

    l1 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="h1",
        invoice_no="INV1",
        invoice_ts=_ts(2010, 3, 10, 9, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=2,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    l2 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="h2",
        invoice_no="INV2",
        invoice_ts=_ts(2010, 7, 20, 15, 0),
        date_key=d2,
        store_id=store.store_id,
        product_id=p1.product_id,
        quantity=4,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    db_session.add_all([l1, l2])
    db_session.flush()

    updated = compute_product_first_last_dates(db_session)
    assert updated == 1

    refreshed_p1 = db_session.get(DimProduct, p1.product_id)
    assert refreshed_p1 is not None
    assert refreshed_p1.first_sale_date == d1
    assert refreshed_p1.last_sale_date == d2


def test_customer_order_dates_computation(db_session: Session) -> None:
    """Verify first_order_date and last_order_date updates on dim_customer."""
    store, p1, _, _ = _seed_basic_dimensions(db_session)
    cust = DimCustomer(customer_id=1, customer_key="cust_hash_123", country="GB")
    db_session.add(cust)
    db_session.flush()

    d1 = datetime.date(2010, 1, 15)
    d2 = datetime.date(2010, 8, 22)
    ensure_dim_date(db_session, d1, d2)

    batch = _create_batch(db_session)

    l1 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="ch1",
        invoice_no="INV1",
        invoice_ts=_ts(2010, 1, 15, 9, 0),
        date_key=d1,
        store_id=store.store_id,
        product_id=p1.product_id,
        customer_id=cust.customer_id,
        quantity=2,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    l2 = FactSalesLine(
        batch_id=batch.batch_id,
        source_row_hash="ch2",
        invoice_no="INV2",
        invoice_ts=_ts(2010, 8, 22, 14, 0),
        date_key=d2,
        store_id=store.store_id,
        product_id=p1.product_id,
        customer_id=cust.customer_id,
        quantity=4,
        unit_price=Decimal("10.00"),
        is_return=False,
        is_outlier=False,
    )
    db_session.add_all([l1, l2])
    db_session.flush()

    updated = compute_customer_order_dates(db_session)
    assert updated == 1

    refreshed_c = db_session.get(DimCustomer, cust.customer_id)
    assert refreshed_c is not None
    assert refreshed_c.first_order_date == d1
    assert refreshed_c.last_order_date == d2


def test_abc_classification(db_session: Session) -> None:
    """Verify ABC classification logic on trailing 365 days revenue."""
    store, p1, p2, p_non = _seed_basic_dimensions(db_session)
    p3 = DimProduct(product_id=4, sku="SKU003", description="Product Three", is_product=True)
    p_zero = DimProduct(product_id=5, sku="SKU004", description="Zero Sales", is_product=True)
    db_session.add_all([p3, p_zero])
    db_session.flush()

    as_of = datetime.date(2011, 12, 1)
    ensure_dim_date(db_session, as_of - datetime.timedelta(days=365), as_of)

    # Revenue proportions: P1 = 800 (80%), P2 = 150 (15%), P3 = 50 (5%), Total = 1000
    r1 = FactDailySales(
        date_key=as_of,
        store_id=store.store_id,
        product_id=p1.product_id,
        units_sold=80,
        units_returned=0,
        revenue=Decimal("800.00"),
        n_orders=10,
    )
    r2 = FactDailySales(
        date_key=as_of,
        store_id=store.store_id,
        product_id=p2.product_id,
        units_sold=15,
        units_returned=0,
        revenue=Decimal("150.00"),
        n_orders=5,
    )
    r3 = FactDailySales(
        date_key=as_of,
        store_id=store.store_id,
        product_id=p3.product_id,
        units_sold=5,
        units_returned=0,
        revenue=Decimal("50.00"),
        n_orders=2,
    )
    db_session.add_all([r1, r2, r3])
    db_session.flush()

    counts = classify_abc_xyz(db_session, as_of=as_of)

    p1_ref = db_session.get(DimProduct, p1.product_id)
    p2_ref = db_session.get(DimProduct, p2.product_id)
    p3_ref = db_session.get(DimProduct, p3.product_id)
    p_zero_ref = db_session.get(DimProduct, p_zero.product_id)
    p_non_ref = db_session.get(DimProduct, p_non.product_id)

    assert p1_ref is not None
    assert p1_ref.abc_class == "A"
    assert p2_ref is not None
    assert p2_ref.abc_class == "B"
    assert p3_ref is not None
    assert p3_ref.abc_class == "C"
    assert p_zero_ref is not None
    assert p_zero_ref.abc_class == "C"
    assert p_non_ref is not None
    assert p_non_ref.abc_class is None

    assert counts["A"] == 1
    assert counts["B"] == 1
    assert counts["C"] == 2


def test_xyz_classification(db_session: Session) -> None:
    """Verify XYZ classification based on weekly demand coefficient of variation."""
    store = DimStore(store_id=1, store_code="ONLINE", store_name="Online Store", country="GB")
    # p_x: constant weekly demand (CV = 0 < 0.5)
    p_x = DimProduct(product_id=1, sku="SKU_X", description="Stable Demand", is_product=True)
    # p_y: moderate variation
    p_y = DimProduct(product_id=2, sku="SKU_Y", description="Moderate Demand", is_product=True)
    # p_z: erratic / intermittent demand (only 1 sale in all weeks)
    p_z = DimProduct(product_id=3, sku="SKU_Z", description="Erratic Demand", is_product=True)
    # p_zero: zero sales -> Z
    p_zero = DimProduct(product_id=4, sku="SKU_ZERO", description="Zero Demand", is_product=True)
    # p_non: non-product -> None
    p_non = DimProduct(product_id=5, sku="POST", description="Fee", is_product=False)

    db_session.add_all([store, p_x, p_y, p_z, p_zero, p_non])
    db_session.flush()

    start_date = datetime.date(2011, 1, 3)  # Monday
    end_date = datetime.date(2011, 3, 27)  # 12 weeks
    ensure_dim_date(db_session, start_date, end_date)

    # Populate weekly sales for 12 weeks
    daily_records: list[FactDailySales] = []
    current_monday = start_date
    week_idx = 0
    while current_monday <= end_date:
        # P_X: always 10 units each week
        daily_records.append(
            FactDailySales(
                date_key=current_monday,
                store_id=store.store_id,
                product_id=p_x.product_id,
                units_sold=10,
                revenue=Decimal("100.00"),
                n_orders=1,
            )
        )
        # P_Y: alternating weekly demand between 5 and 25 units
        # mean 15, std ~ 10.44 -> CV = 10.44 / 15 = 0.696 (class Y)
        y_qty = 25 if week_idx % 2 == 0 else 5
        daily_records.append(
            FactDailySales(
                date_key=current_monday,
                store_id=store.store_id,
                product_id=p_y.product_id,
                units_sold=y_qty,
                revenue=Decimal(str(y_qty * 10)),
                n_orders=1,
            )
        )
        # P_Z: only sells in week 0
        if week_idx == 0:
            daily_records.append(
                FactDailySales(
                    date_key=current_monday,
                    store_id=store.store_id,
                    product_id=p_z.product_id,
                    units_sold=10,
                    revenue=Decimal("100.00"),
                    n_orders=1,
                )
            )

        current_monday += datetime.timedelta(days=7)
        week_idx += 1

    db_session.add_all(daily_records)
    db_session.flush()

    counts = classify_abc_xyz(db_session, as_of=end_date)

    p_x_ref = db_session.get(DimProduct, p_x.product_id)
    p_y_ref = db_session.get(DimProduct, p_y.product_id)
    p_z_ref = db_session.get(DimProduct, p_z.product_id)
    p_zero_ref = db_session.get(DimProduct, p_zero.product_id)
    p_non_ref = db_session.get(DimProduct, p_non.product_id)

    assert p_x_ref is not None
    assert p_x_ref.xyz_class == "X"
    assert p_y_ref is not None
    assert p_y_ref.xyz_class == "Y"
    assert p_z_ref is not None
    assert p_z_ref.xyz_class == "Z"
    assert p_zero_ref is not None
    assert p_zero_ref.xyz_class == "Z"
    assert p_non_ref is not None
    assert p_non_ref.xyz_class is None

    assert counts["X"] == 1
    assert counts["Y"] == 1
    assert counts["Z"] == 2


def test_reconciliation_on_mini_retail_fixture(db_session: Session) -> None:
    """VERIFY P1-09 RECONCILIATION:

    Reconciliation test:
    sum(units_sold) == sum(positive, non-outlier product lines on complete days)
    """
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    assert fixture_path.exists(), "mini_retail.csv fixture must exist"

    # 1. Load the fixture using the canonical ingestion loader
    adapter = OnlineRetailIIAdapter()
    raw_df = adapter.read(fixture_path)
    cleaned = clean_sales_lines(raw_df, salt=TEST_SALT)

    # 2. Check and mark partial last day as incomplete per TRD §4.4
    last_date, is_complete = evaluate_dataset_last_day_completeness(cleaned.df["invoice_ts"])
    if not is_complete:
        ensure_dim_date(db_session, last_date, last_date)
        mark_dim_date_completeness(db_session, last_date, is_complete=False)

    load_res = load_sales_dataset(
        db_session,
        cleaned,
        source_name="online_retail_ii",
        source_file=str(fixture_path),
    )
    assert load_res.status == "succeeded"
    assert load_res.rows_loaded > 0

    # Ensure completeness status is maintained on dim_date for the last day
    if not is_complete:
        mark_dim_date_completeness(db_session, last_date, is_complete=False)

    # 3. Run the daily sales aggregation
    written = aggregate_daily_sales(db_session)
    assert written > 0

    # 4. Compute expected sum of units sold directly from FactSalesLine:
    expected_stmt = (
        sa.select(sa.func.coalesce(sa.func.sum(FactSalesLine.quantity), 0))
        .join(DimProduct, FactSalesLine.product_id == DimProduct.product_id)
        .join(DimDate, FactSalesLine.date_key == DimDate.date_key)
        .where(
            FactSalesLine.is_return.is_(False),
            FactSalesLine.quantity > 0,
            FactSalesLine.is_outlier.is_(False),
            DimProduct.is_product.is_(True),
            DimDate.is_complete.is_(True),
        )
    )
    expected_units_sold = db_session.scalar(expected_stmt) or 0

    # 5. Compute actual sum of units sold in FactDailySales
    actual_stmt = sa.select(sa.func.coalesce(sa.func.sum(FactDailySales.units_sold), 0))
    actual_units_sold = db_session.scalar(actual_stmt) or 0

    # 6. EXACT RECONCILIATION MATCH
    assert actual_units_sold > 0
    assert actual_units_sold == expected_units_sold, (
        f"Reconciliation failure: FactDailySales.units_sold ({actual_units_sold}) "
        f"!= expected positive non-outlier complete-day product sales ({expected_units_sold})"
    )

    # Also verify full aggregation pipeline runs cleanly on the fixture
    agg_res = run_aggregation_pipeline(db_session)
    assert agg_res.daily_records_written == written
    assert agg_res.products_updated >= 0
    assert sum(agg_res.abc_counts.values()) > 0
    assert sum(agg_res.xyz_counts.values()) > 0
