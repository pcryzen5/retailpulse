"""Ingestion adapter for Generic CSV uploads per TRD §4.1."""

from __future__ import annotations

import os
from typing import BinaryIO, TextIO

import numpy as np
import pandas as pd

from retailpulse.errors import IngestionError
from retailpulse.ingestion.schemas import (
    GENERIC_CSV_REQUIRED,
    canonicalize_customer_id,
    canonicalize_invoice_no,
    canonicalize_stock_code,
    normalize_column_name,
    validate_required_columns,
)
from retailpulse.logging import get_logger

logger = get_logger(__name__)

GENERIC_COLUMN_ALIASES: dict[str, str] = {
    "invoiceno": "invoice_no",
    "invoice_no": "invoice_no",
    "invoice": "invoice_no",
    "invoicedate": "invoice_date",
    "invoice_date": "invoice_date",
    "invoicets": "invoice_date",
    "invoice_ts": "invoice_date",
    "sku": "sku",
    "stockcode": "sku",
    "stock_code": "sku",
    "quantity": "quantity",
    "qty": "quantity",
    "unitprice": "unit_price",
    "unit_price": "unit_price",
    "price": "unit_price",
    "customerid": "customer_id",
    "customer_id": "customer_id",
    "storecode": "store_code",
    "store_code": "store_code",
    "store": "store_code",
    "description": "description",
    "desc": "description",
    "country": "country",
    "category": "category",
}


def _normalize_generic_columns(columns: list[str]) -> dict[str, str]:
    """Map raw columns to standardized generic CSV column names."""
    return {col: GENERIC_COLUMN_ALIASES.get(normalize_column_name(col), col) for col in columns}


class GenericCSVAdapter:
    """Ingestion adapter for Generic CSV sales line uploads."""

    @classmethod
    def read_csv(
        cls,
        source: str | os.PathLike[str] | BinaryIO | TextIO,
        default_store_code: str = "ONLINE",
        default_country: str = "GB",
        **pandas_kwargs: object,
    ) -> pd.DataFrame:
        """Read a Generic CSV and produce the canonical sales line DataFrame.

        Required: invoice_no, invoice_date, sku, quantity, unit_price
        Optional: customer_id, store_code (default ONLINE), description, country (default GB)
        """
        logger.info("reading generic csv", source=str(source))
        try:
            raw_df = pd.read_csv(
                source,
                dtype=str,
                keep_default_na=False,
                encoding="utf-8",
                **pandas_kwargs,
            )
        except Exception as exc:
            raise IngestionError(
                f"Failed to read Generic CSV from {source}: {exc}",
                details={"source": str(source), "error": str(exc)},
            ) from exc

        if raw_df.empty:
            return pd.DataFrame(
                columns=[
                    "invoice_no",
                    "invoice_ts",
                    "sku",
                    "description",
                    "quantity",
                    "unit_price",
                    "customer_id",
                    "country",
                    "store_code",
                    "sheet_name",
                    "sheet_row_idx",
                ]
            )

        # 1. Rename columns by aliases
        rename_map = _normalize_generic_columns(list(raw_df.columns))
        df = raw_df.rename(columns=rename_map)

        # 2. Validate required columns
        validate_required_columns(
            present_columns=set(df.columns),
            required_columns=GENERIC_CSV_REQUIRED,
            source_name="Generic CSV",
        )

        # 3. Canonical field mapping
        invoice_no = df["invoice_no"].apply(canonicalize_invoice_no)
        invoice_ts = pd.to_datetime(df["invoice_date"], errors="coerce")
        sku = df["sku"].apply(canonicalize_stock_code)

        description = (
            df["description"]
            .astype(str)
            .str.strip()
            .replace({"": None, "nan": None, "None": None, "NaN": None})
            if "description" in df.columns
            else pd.Series([None] * len(df))
        )

        quantity = pd.to_numeric(df["quantity"], errors="coerce")
        unit_price = pd.to_numeric(df["unit_price"], errors="coerce")

        customer_id = (
            df["customer_id"].apply(canonicalize_customer_id)
            if "customer_id" in df.columns
            else pd.Series([None] * len(df))
        )

        country = (
            df["country"]
            .astype(str)
            .str.strip()
            .replace({"": default_country, "nan": default_country})
            if "country" in df.columns
            else pd.Series([default_country] * len(df))
        )

        store_code = (
            df["store_code"]
            .astype(str)
            .str.strip()
            .replace({"": default_store_code, "nan": default_store_code})
            if "store_code" in df.columns
            else pd.Series([default_store_code] * len(df))
        )

        canonical = pd.DataFrame(
            {
                "invoice_no": invoice_no,
                "invoice_ts": invoice_ts,
                "sku": sku,
                "description": description,
                "quantity": quantity,
                "unit_price": unit_price,
                "customer_id": customer_id,
                "country": country,
                "store_code": store_code,
                "sheet_name": "default",
                "sheet_row_idx": np.arange(len(df), dtype=int),
            }
        )

        return canonical

    @classmethod
    def read(
        cls,
        source: str | os.PathLike[str] | BinaryIO | TextIO,
        default_store_code: str = "ONLINE",
    ) -> pd.DataFrame:
        """Alias for read_csv."""
        return cls.read_csv(source, default_store_code=default_store_code)
