"""Canonical ingestion schemas, column mappings, and normalization helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from retailpulse.errors import IngestionError

# Canonical sales line column names per TRD §4.2
CANONICAL_COLUMNS: list[str] = [
    "invoice_no",
    "invoice_ts",
    "sku",
    "description",
    "quantity",
    "unit_price",
    "customer_id",
    "country",
    "store_code",
]

# Online Retail II aliases mapped to canonical intermediate names
ONLINE_RETAIL_II_ALIASES: dict[str, str] = {
    # Invoice aliases
    "invoice": "Invoice",
    "invoiceno": "Invoice",
    "invoice_no": "Invoice",
    "invoicenum": "Invoice",
    # StockCode aliases
    "stockcode": "StockCode",
    "stock_code": "StockCode",
    "sku": "StockCode",
    "productcode": "StockCode",
    # Description aliases
    "description": "Description",
    "desc": "Description",
    # Quantity aliases
    "quantity": "Quantity",
    "qty": "Quantity",
    # InvoiceDate aliases
    "invoicedate": "InvoiceDate",
    "invoice_date": "InvoiceDate",
    "invoicets": "InvoiceDate",
    "invoice_ts": "InvoiceDate",
    # Price aliases
    "price": "Price",
    "unitprice": "Price",
    "unit_price": "Price",
    # Customer ID aliases
    "customerid": "Customer ID",
    "customer_id": "Customer ID",
    "customer id": "Customer ID",
    "customerno": "Customer ID",
    # Country aliases
    "country": "Country",
}

# Required columns for Online Retail II
ONLINE_RETAIL_II_REQUIRED: set[str] = {
    "Invoice",
    "StockCode",
    "Quantity",
    "InvoiceDate",
    "Price",
    "Country",
}

# Generic CSV required and optional columns per TRD §4.1
GENERIC_CSV_REQUIRED: set[str] = {
    "invoice_no",
    "invoice_date",
    "sku",
    "quantity",
    "unit_price",
}


def normalize_column_name(col: str) -> str:
    """Normalize a raw column name by stripping whitespace, lowercasing, and removing separators."""
    return col.strip().lower().replace(" ", "").replace("_", "")


def canonicalize_customer_id(val: Any) -> str | None:
    """Canonicalize raw Customer ID into clean integer text without decimal parts.

    Per TRD §4.3 rule 1:
    "canonicalize Customer ID to integer text with no decimal part (the workbook stores it as
    a number with blanks, so a naive string conversion gives 13085.0)"
    """
    if val is None or pd.isna(val):
        return None

    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None

    try:
        num = float(s)
        if np.isfinite(num) and num.is_integer():
            return str(int(num))
    except (ValueError, OverflowError):
        pass

    if s.endswith(".0"):
        s = s[:-2]

    return s if s else None


def canonicalize_stock_code(val: Any) -> str:
    """Normalize StockCode / SKU by trimming whitespace and uppercasing.

    Per TRD §4.3 rule 1:
    "normalize StockCode (strip, uppercase; 173 codes in the real file differ only by case)"
    """
    if val is None or pd.isna(val):
        return ""
    return str(val).strip().upper()


def canonicalize_invoice_no(val: Any) -> str:
    """Normalize invoice number by reading as text and trimming whitespace."""
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    # In Excel, a numeric invoice e.g. 536365 might be read as 536365.0
    if s.endswith(".0"):
        try:
            num = float(s)
            if num.is_integer():
                return str(int(num))
        except (ValueError, OverflowError):
            pass
    return s


def validate_required_columns(
    present_columns: set[str],
    required_columns: set[str],
    source_name: str,
) -> None:
    """Raise IngestionError if any required column is missing."""
    missing = required_columns - present_columns
    if missing:
        raise IngestionError(
            f"{source_name} is missing required column(s): {sorted(missing)}. "
            f"Present columns: {sorted(present_columns)}",
            details={
                "missing_columns": sorted(missing),
                "present_columns": sorted(present_columns),
                "source": source_name,
            },
        )
