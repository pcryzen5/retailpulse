"""Data ingestion loaders for dimensions, sales lines, and batch tracking.

Implements TRD §4.3 idempotent database loading:
- Dimension upserts (core.dim_store, core.dim_product, core.dim_customer)
- Chunked sales lines loading into core.fact_sales_line with ON CONFLICT DO NOTHING
- Date dimension verification in core.dim_date
- Ingestion batch registration and status tracking in ops.ingestion_batch
- Quality check findings persistence in ops.data_quality_result
"""

from __future__ import annotations

import datetime
import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from retailpulse.db.models.core import DimCustomer, DimDate, DimProduct, DimStore, FactSalesLine
from retailpulse.db.models.ops import DataQualityResult, IngestionBatch
from retailpulse.logging import get_logger
from retailpulse.timeutil import utc_now

logger = get_logger(__name__)

# Default chunk size for bulk database operations
DEFAULT_CHUNK_SIZE: int = 10_000


@dataclass(frozen=True)
class LoadResult:
    """Summary of an ingestion batch load execution."""

    batch_id: uuid.UUID
    rows_in: int
    rows_loaded: int
    rows_rejected: int
    rows_skipped: int
    products_upserted: int
    customers_upserted: int
    status: str


def compute_file_sha256(path: Path | str) -> str:
    """Compute sha256 checksum of a file on disk in 64KB blocks.

    Args:
        path: Path to the target file.

    Returns:
        Hexadecimal 64-character lowercase sha256 digest string.
    """
    hasher = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as f:
        while chunk := f.read(65_536):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dates_exist(session: Session, dates: set[datetime.date]) -> int:
    """Ensure core.dim_date has records for all specified calendar dates.

    Args:
        session: Active database session.
        dates: Set of dates to verify and populate if missing.

    Returns:
        Count of newly inserted date records.
    """
    if not dates:
        return 0

    existing_stmt = select(DimDate.date_key).where(DimDate.date_key.in_(dates))
    existing_dates = set(session.execute(existing_stmt).scalars().all())
    missing_dates = dates - existing_dates

    if not missing_dates:
        return 0

    new_records = [
        DimDate(
            date_key=d,
            year=d.year,
            quarter=(d.month - 1) // 3 + 1,
            month=d.month,
            iso_week=int(d.isocalendar()[1]),
            day_of_week=d.isoweekday(),
            day_of_month=d.day,
            is_weekend=d.isoweekday() in (6, 7),
            is_holiday=False,
            holiday_name=None,
            is_trading_day=d.isoweekday() != 6,
            is_complete=True,
        )
        for d in sorted(missing_dates)
    ]
    session.add_all(new_records)
    session.flush()
    logger.info("dim_date records populated", count=len(new_records))
    return len(new_records)


def upsert_stores(
    session: Session,
    store_codes: set[str] | Sequence[str] | None = None,
) -> dict[str, int]:
    """Ensure stores exist in core.dim_store and return code-to-id mapping.

    Args:
        session: Active database session.
        store_codes: Iterable of store code strings (defaults to {'ONLINE'}).

    Returns:
        Dictionary mapping store_code -> store_id.
    """
    codes = set(store_codes) if store_codes else {"ONLINE"}
    stmt = select(DimStore).where(DimStore.store_code.in_(codes))
    existing = {s.store_code: s for s in session.execute(stmt).scalars().all()}

    for code in codes:
        if code not in existing:
            name = "Online store" if code == "ONLINE" else f"Store {code}"
            new_store = DimStore(store_code=code, store_name=name, country="GB")
            session.add(new_store)
            existing[code] = new_store

    session.flush()
    return {code: s.store_id for code, s in existing.items()}


def upsert_products(session: Session, df: pd.DataFrame) -> tuple[dict[str, int], int]:
    """Upsert products into core.dim_product with date range expansion.

    Args:
        session: Active database session.
        df: Cleaned sales DataFrame with columns 'sku', 'invoice_ts', etc.

    Returns:
        Tuple of ({sku: product_id}, count_of_newly_inserted_products).
    """
    if df.empty or "sku" not in df.columns:
        return {}, 0

    meta = _extract_product_metadata(df)
    stmt = select(DimProduct).where(DimProduct.sku.in_(set(meta.keys())))
    existing_map = {p.sku: p for p in session.execute(stmt).scalars().all()}

    new_count = 0
    for sku, info in meta.items():
        if sku in existing_map:
            _update_product_record(existing_map[sku], info)
        else:
            new_prod = DimProduct(
                sku=sku,
                description=info["description"],
                is_product=info["is_product"],
                first_sale_date=info["first_date"],
                last_sale_date=info["last_date"],
            )
            session.add(new_prod)
            existing_map[sku] = new_prod
            new_count += 1

    session.flush()
    return {sku: p.product_id for sku, p in existing_map.items()}, new_count


def _extract_product_metadata(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Extract aggregated per-SKU date ranges and descriptions from frame."""
    meta: dict[str, dict[str, Any]] = {}
    has_prod_col = "is_product" in df.columns
    has_desc = "description" in df.columns

    for sku, grp in df.groupby("sku", sort=False):
        sku_str = str(sku)
        ts_series = pd.to_datetime(grp["invoice_ts"])
        dates = ts_series.dt.date
        desc = None
        if has_desc:
            non_null = grp["description"].dropna()
            desc = str(non_null.iloc[-1]).strip() if not non_null.empty else None

        is_prod = bool(grp["is_product"].iloc[-1]) if has_prod_col else True
        meta[sku_str] = {
            "first_date": dates.min(),
            "last_date": dates.max(),
            "description": desc,
            "is_product": is_prod,
        }
    return meta


def _update_product_record(record: DimProduct, info: dict[str, Any]) -> None:
    """Update date ranges and missing description on an existing product."""
    f_date = info["first_date"]
    l_date = info["last_date"]
    if f_date and (record.first_sale_date is None or f_date < record.first_sale_date):
        record.first_sale_date = f_date
    if l_date and (record.last_sale_date is None or l_date > record.last_sale_date):
        record.last_sale_date = l_date
    if record.description is None and info["description"] is not None:
        record.description = info["description"]


def upsert_customers(session: Session, df: pd.DataFrame) -> tuple[dict[str, int], int]:
    """Upsert customers into core.dim_customer with order date expansion.

    Args:
        session: Active database session.
        df: Cleaned sales DataFrame with columns 'customer_key', 'invoice_ts', 'country'.

    Returns:
        Tuple of ({customer_key: customer_id}, count_of_newly_inserted_customers).
    """
    if df.empty or "customer_key" not in df.columns:
        return {}, 0

    cust_df = df[df["customer_key"].notna() & (df["customer_key"] != "")]
    if cust_df.empty:
        return {}, 0

    meta = _extract_customer_metadata(cust_df)
    stmt = select(DimCustomer).where(DimCustomer.customer_key.in_(set(meta.keys())))
    existing_map = {c.customer_key: c for c in session.execute(stmt).scalars().all()}

    new_count = 0
    for key, info in meta.items():
        if key in existing_map:
            _update_customer_record(existing_map[key], info)
        else:
            new_cust = DimCustomer(
                customer_key=key,
                country=info["country"],
                first_order_date=info["first_date"],
                last_order_date=info["last_date"],
            )
            session.add(new_cust)
            existing_map[key] = new_cust
            new_count += 1

    session.flush()
    return {key: c.customer_id for key, c in existing_map.items()}, new_count


def _extract_customer_metadata(cust_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Extract per-customer order dates and primary country from frame."""
    meta: dict[str, dict[str, Any]] = {}
    has_country = "country" in cust_df.columns

    for key, grp in cust_df.groupby("customer_key", sort=False):
        key_str = str(key)
        ts_series = pd.to_datetime(grp["invoice_ts"])
        dates = ts_series.dt.date
        country = None
        if has_country:
            non_null = grp["country"].dropna()
            country = str(non_null.iloc[-1]).strip() if not non_null.empty else None

        meta[key_str] = {
            "first_date": dates.min(),
            "last_date": dates.max(),
            "country": country,
        }
    return meta


def _update_customer_record(record: DimCustomer, info: dict[str, Any]) -> None:
    """Update order dates and missing country on an existing customer."""
    f_date = info["first_date"]
    l_date = info["last_date"]
    if f_date and (record.first_order_date is None or f_date < record.first_order_date):
        record.first_order_date = f_date
    if l_date and (record.last_order_date is None or l_date > record.last_order_date):
        record.last_order_date = l_date
    if record.country is None and info["country"] is not None:
        record.country = info["country"]


def insert_sales_lines_chunked(
    session: Session,
    rows: list[dict[str, Any]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> int:
    """Insert sales lines in chunks using ON CONFLICT (source_row_hash) DO NOTHING.

    Args:
        session: Active database session.
        rows: Dictionaries representing FactSalesLine rows to insert.
        chunk_size: Maximum row count per bulk statement.

    Returns:
        Number of input rows processed.
    """
    if not rows:
        return 0

    bind = session.get_bind()
    dialect_name = bind.dialect.name if bind is not None else "postgresql"
    dialect_insert = sqlite_insert if dialect_name == "sqlite" else pg_insert

    # SQLite limits host parameters to 999 (13 columns per row -> max 70 rows)
    step = min(chunk_size, 50) if dialect_name == "sqlite" else chunk_size

    for i in range(0, len(rows), step):
        chunk = rows[i : i + step]
        stmt = (
            dialect_insert(FactSalesLine)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["source_row_hash"])
        )
        session.execute(stmt)

    session.flush()
    return len(rows)


def record_quality_findings(
    session: Session,
    batch_id: uuid.UUID,
    findings: Sequence[Any],
) -> int:
    """Persist data quality validation findings into ops.data_quality_result.

    Args:
        session: Active database session.
        batch_id: Ingestion batch identifier.
        findings: Sequence of QualityFinding instances or dictionaries.

    Returns:
        Count of recorded quality findings.
    """
    if not findings:
        return 0

    existing_stmt = select(DataQualityResult.check_name).where(
        DataQualityResult.batch_id == batch_id
    )
    existing_checks = set(session.execute(existing_stmt).scalars().all())

    inserted_count = 0
    for finding in findings:
        name = getattr(finding, "check_name", None) or (
            finding.get("check_name") if isinstance(finding, dict) else None
        )
        if not name or name in existing_checks:
            continue

        severity = getattr(finding, "severity", None) or (
            finding.get("severity", "info") if isinstance(finding, dict) else "info"
        )
        passed = bool(
            getattr(finding, "passed", None)
            if hasattr(finding, "passed")
            else (finding.get("passed", True) if isinstance(finding, dict) else True)
        )
        affected = int(
            getattr(finding, "affected_rows", None)
            or (finding.get("affected_rows", 0) if isinstance(finding, dict) else 0)
        )
        details = getattr(finding, "details", None) or (
            finding.get("details", {}) if isinstance(finding, dict) else {}
        )

        result = DataQualityResult(
            batch_id=batch_id,
            check_name=str(name),
            severity=str(severity),
            passed=passed,
            affected_rows=affected,
            details=details,
        )
        session.add(result)
        existing_checks.add(str(name))
        inserted_count += 1

    session.flush()
    return inserted_count


def get_or_create_batch(
    session: Session,
    *,
    source_name: str,
    source_file: str,
    file_sha256: str,
    original_filename: str | None = None,
    batch_id: uuid.UUID | None = None,
    row_count_in: int | None = None,
) -> tuple[IngestionBatch, bool]:
    """Fetch existing batch by (source_name, file_sha256) or create a new one.

    Args:
        session: Active database session.
        source_name: Data source identifier.
        source_file: Stored file path or reference.
        file_sha256: 64-character sha256 checksum.
        original_filename: Original file name.
        batch_id: Optional explicit UUID to assign if creating.
        row_count_in: Input raw row count.

    Returns:
        Tuple of (IngestionBatch, is_new_boolean).
    """
    if batch_id is not None:
        batch = session.get(IngestionBatch, batch_id)
        if batch is not None:
            return batch, False

    stmt = select(IngestionBatch).where(
        IngestionBatch.source_name == source_name,
        IngestionBatch.file_sha256 == file_sha256,
    )
    existing = session.execute(stmt).scalar_one_or_none()
    if existing is not None:
        return existing, False

    now = utc_now()
    new_batch = IngestionBatch(
        batch_id=batch_id or uuid.uuid4(),
        source_name=source_name,
        source_file=source_file,
        original_filename=original_filename,
        file_sha256=file_sha256,
        row_count_in=row_count_in,
        status="running",
        started_at=now,
    )
    session.add(new_batch)
    session.flush()
    return new_batch, True


def _prepare_sales_line_rows(
    df: pd.DataFrame,
    batch_id: uuid.UUID,
    store_map: dict[str, int],
    prod_map: dict[str, int],
    cust_map: dict[str, int],
) -> list[dict[str, Any]]:
    """Convert cleaned DataFrame records into FactSalesLine insert dictionaries."""
    rows: list[dict[str, Any]] = []
    has_store = "store_code" in df.columns
    has_cust = "customer_key" in df.columns
    has_return = "is_return" in df.columns
    has_outlier = "is_outlier" in df.columns

    for row in df.itertuples(index=False):
        inv_no = str(row.invoice_no)
        ts = pd.to_datetime(row.invoice_ts).to_pydatetime()
        d_key = ts.date()

        store_code = getattr(row, "store_code", "ONLINE") if has_store else "ONLINE"
        store_id = store_map.get(str(store_code), store_map.get("ONLINE", 1))

        sku = str(row.sku)
        product_id = prod_map[sku]

        customer_id = None
        if has_cust:
            ck = getattr(row, "customer_key", None)
            if ck and pd.notna(ck):
                customer_id = cust_map.get(str(ck))

        qty = int(row.quantity)
        price = Decimal(str(round(float(row.unit_price), 4)))
        is_ret = bool(getattr(row, "is_return", False)) if has_return else qty < 0
        is_out = bool(getattr(row, "is_outlier", False)) if has_outlier else False

        rows.append(
            {
                "batch_id": batch_id,
                "source_row_hash": str(row.source_row_hash),
                "invoice_no": inv_no,
                "invoice_ts": ts,
                "date_key": d_key,
                "store_id": store_id,
                "product_id": product_id,
                "customer_id": customer_id,
                "quantity": qty,
                "unit_price": price,
                "is_return": is_ret,
                "is_outlier": is_out,
                "currency": "GBP",
            }
        )
    return rows


def load_sales_dataset(
    session: Session,
    dataset: Any,
    *,
    source_name: str,
    source_file: str,
    file_sha256: str | None = None,
    original_filename: str | None = None,
    batch_id: uuid.UUID | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> LoadResult:
    """Execute end-to-end idempotent loading of a cleaned sales dataset.

    Args:
        session: Active database session.
        dataset: CleanedDataset or pandas DataFrame with cleaned records.
        source_name: Source identifier (e.g. 'online_retail_ii').
        source_file: Source file path or logical storage reference.
        file_sha256: Optional precomputed sha256 checksum of source file.
        original_filename: Optional original filename.
        batch_id: Optional explicit batch UUID.
        chunk_size: Row batch size for chunked database operations.

    Returns:
        LoadResult summary.
    """
    df = getattr(dataset, "df", dataset)
    findings = getattr(dataset, "findings", [])
    row_counts = getattr(dataset, "row_counts", {})
    quarantined = getattr(dataset, "quarantined_df", None)

    sha256_val = _resolve_file_sha256(source_file, file_sha256, df)
    rows_in = row_counts.get("raw", len(df) + (len(quarantined) if quarantined is not None else 0))

    batch, _ = get_or_create_batch(
        session,
        source_name=source_name,
        source_file=source_file,
        file_sha256=sha256_val,
        original_filename=original_filename,
        batch_id=batch_id,
        row_count_in=rows_in,
    )

    record_quality_findings(session, batch.batch_id, findings)

    if _is_batch_rejected(findings):
        return _handle_rejected_batch(session, batch, rows_in)

    return _execute_data_load(
        session,
        batch=batch,
        df=df,
        rows_in=rows_in,
        quarantined=quarantined,
        row_counts=row_counts,
        chunk_size=chunk_size,
    )


def _resolve_file_sha256(source_file: str, file_sha256: str | None, df: pd.DataFrame) -> str:
    """Resolve or calculate sha256 checksum of the target data."""
    if file_sha256:
        return file_sha256
    file_path = Path(source_file)
    if file_path.is_file():
        return compute_file_sha256(file_path)
    return hashlib.sha256(f"{source_file}:{len(df)}".encode()).hexdigest()


def _is_batch_rejected(findings: Sequence[Any]) -> bool:
    """Check if any finding indicates the batch rejection gate failed."""
    for f in findings:
        name = getattr(f, "check_name", None) or (
            f.get("check_name") if isinstance(f, dict) else ""
        )
        passed = (
            getattr(f, "passed", True)
            if hasattr(f, "passed")
            else (f.get("passed", True) if isinstance(f, dict) else True)
        )
        if name == "batch_rejection_gate" and not passed:
            return True
    return False


def _handle_rejected_batch(session: Session, batch: IngestionBatch, rows_in: int) -> LoadResult:
    """Record rejection status and return zero-load result."""
    batch.status = "rejected"
    batch.row_count_in = rows_in
    batch.row_count_loaded = 0
    batch.row_count_rejected = rows_in
    batch.finished_at = utc_now()
    session.flush()
    return LoadResult(
        batch_id=batch.batch_id,
        rows_in=rows_in,
        rows_loaded=0,
        rows_rejected=rows_in,
        rows_skipped=0,
        products_upserted=0,
        customers_upserted=0,
        status="rejected",
    )


def _execute_data_load(
    session: Session,
    *,
    batch: IngestionBatch,
    df: pd.DataFrame,
    rows_in: int,
    quarantined: pd.DataFrame | None,
    row_counts: dict[str, int],
    chunk_size: int,
) -> LoadResult:
    """Perform dimension upserts and sales lines chunked loading."""
    stores = set(df["store_code"].dropna().unique()) if "store_code" in df.columns else {"ONLINE"}
    store_map = upsert_stores(session, stores)
    prod_map, new_prods = upsert_products(session, df)
    cust_map, new_custs = upsert_customers(session, df)

    dates = set(pd.to_datetime(df["invoice_ts"]).dt.date.unique())
    ensure_dates_exist(session, dates)

    rows = _prepare_sales_line_rows(df, batch.batch_id, store_map, prod_map, cust_map)

    count_stmt = (
        select(func.count())
        .select_from(FactSalesLine)
        .where(FactSalesLine.batch_id == batch.batch_id)
    )
    initial_count = session.scalar(count_stmt) or 0

    insert_sales_lines_chunked(session, rows, chunk_size=chunk_size)

    final_count = session.scalar(count_stmt) or 0
    newly_loaded = final_count - initial_count
    skipped = len(rows) - newly_loaded
    rejected_count = len(quarantined) if quarantined is not None else row_counts.get("rejected", 0)

    batch.row_count_in = rows_in
    batch.row_count_loaded = final_count
    batch.row_count_rejected = rejected_count
    batch.status = "succeeded"
    batch.finished_at = utc_now()
    session.flush()

    return LoadResult(
        batch_id=batch.batch_id,
        rows_in=rows_in,
        rows_loaded=newly_loaded,
        rows_rejected=rejected_count,
        rows_skipped=skipped,
        products_upserted=new_prods,
        customers_upserted=new_custs,
        status="succeeded",
    )
