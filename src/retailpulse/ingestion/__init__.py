"""Data ingestion adapters, schemas, and loaders."""

from retailpulse.ingestion.adapters import GenericCSVAdapter, OnlineRetailIIAdapter
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
    "OnlineRetailIIAdapter",
    "canonicalize_customer_id",
    "canonicalize_invoice_no",
    "canonicalize_stock_code",
]
