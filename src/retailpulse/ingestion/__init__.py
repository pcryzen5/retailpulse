"""Data ingestion adapters, schemas, and loaders."""

from retailpulse.ingestion.adapters import GenericCSVAdapter, OnlineRetailIIAdapter
from retailpulse.ingestion.loaders import (
    LoadResult,
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
from retailpulse.ingestion.schemas import (
    CANONICAL_COLUMNS,
    GENERIC_CSV_REQUIRED,
    ONLINE_RETAIL_II_ALIASES,
    ONLINE_RETAIL_II_REQUIRED,
    canonicalize_customer_id,
    canonicalize_invoice_no,
    canonicalize_stock_code,
)

__all__ = [
    "CANONICAL_COLUMNS",
    "GENERIC_CSV_REQUIRED",
    "ONLINE_RETAIL_II_ALIASES",
    "ONLINE_RETAIL_II_REQUIRED",
    "GenericCSVAdapter",
    "LoadResult",
    "OnlineRetailIIAdapter",
    "canonicalize_customer_id",
    "canonicalize_invoice_no",
    "canonicalize_stock_code",
    "compute_file_sha256",
    "ensure_dates_exist",
    "get_or_create_batch",
    "insert_sales_lines_chunked",
    "load_sales_dataset",
    "record_quality_findings",
    "upsert_customers",
    "upsert_products",
    "upsert_stores",
]
