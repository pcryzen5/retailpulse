"""Ingestion adapter for Online Retail II dataset (CSV and XLSX multi-sheet)."""

from __future__ import annotations

import os
from typing import BinaryIO, TextIO, cast

import numpy as np
import openpyxl
import pandas as pd

from retailpulse.errors import IngestionError
from retailpulse.ingestion.schemas import (
    ONLINE_RETAIL_II_ALIASES,
    ONLINE_RETAIL_II_REQUIRED,
    canonicalize_customer_id,
    canonicalize_invoice_no,
    canonicalize_stock_code,
    normalize_column_name,
    validate_required_columns,
)
from retailpulse.logging import get_logger

logger = get_logger(__name__)


def _normalize_online_retail_columns(columns: list[str]) -> dict[str, str]:
    """Map raw columns to standardized Online Retail II column names using aliases."""
    return {col: ONLINE_RETAIL_II_ALIASES.get(normalize_column_name(col), col) for col in columns}


def _process_raw_frame(
    raw_df: pd.DataFrame,
    sheet_name: str = "default",
    default_store_code: str = "ONLINE",
) -> pd.DataFrame:
    """Normalize raw Online Retail II DataFrame into the canonical sales line schema.

    Canonical columns per TRD §4.2:
    invoice_no, invoice_ts, sku, description, quantity, unit_price, customer_id, country, store_code
    Plus provenance metadata: sheet_name, sheet_row_idx
    """
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
    rename_map = _normalize_online_retail_columns(list(raw_df.columns))
    df = raw_df.rename(columns=rename_map)

    # 2. Validate required columns
    validate_required_columns(
        present_columns=set(df.columns),
        required_columns=ONLINE_RETAIL_II_REQUIRED,
        source_name=f"Online Retail II [{sheet_name}]",
    )

    # 3. Read and normalize fields
    invoice_no = df["Invoice"].apply(canonicalize_invoice_no)
    invoice_ts = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    sku = df["StockCode"].apply(canonicalize_stock_code)

    # Description (nullable string)
    description = (
        df["Description"]
        .astype(str)
        .str.strip()
        .replace({"": None, "nan": None, "None": None, "NaN": None})
    )

    # Quantity & Price (numeric coercion)
    quantity = pd.to_numeric(df["Quantity"], errors="coerce")
    unit_price = pd.to_numeric(df["Price"], errors="coerce")

    # Customer ID (canonical integer text or None)
    customer_id = (
        df["Customer ID"].apply(canonicalize_customer_id)
        if "Customer ID" in df.columns
        else pd.Series([None] * len(df))
    )

    # Country
    country = (
        df["Country"]
        .astype(str)
        .str.strip()
        .replace({"": "United Kingdom", "nan": "United Kingdom"})
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
            "store_code": default_store_code,
            "sheet_name": sheet_name,
            "sheet_row_idx": np.arange(len(df), dtype=int),
        }
    )

    return canonical


class OnlineRetailIIAdapter:
    """Ingestion adapter for UCI Online Retail II dataset in CSV or multi-sheet XLSX format."""

    @classmethod
    def read_csv(
        cls,
        source: str | os.PathLike[str] | BinaryIO | TextIO,
        sheet_name: str = "default",
        default_store_code: str = "ONLINE",
        **pandas_kwargs: object,
    ) -> pd.DataFrame:
        """Read a CSV export of Online Retail II and return the canonical sales line DataFrame."""
        logger.info("reading online retail ii csv", source=str(source), sheet=sheet_name)
        try:
            # Read all columns as string to prevent premature truncation of Invoice/StockCode
            df = pd.read_csv(
                source,
                dtype=str,
                keep_default_na=False,
                **pandas_kwargs,
            )
        except Exception as exc:
            raise IngestionError(
                f"Failed to read Online Retail II CSV from {source}: {exc}",
                details={"source": str(source), "error": str(exc)},
            ) from exc

        return _process_raw_frame(df, sheet_name=sheet_name, default_store_code=default_store_code)

    @classmethod
    def read_xlsx(
        cls,
        source: str | os.PathLike[str] | BinaryIO,
        default_store_code: str = "ONLINE",
    ) -> pd.DataFrame:
        """Read all sheets from an Online Retail II Excel workbook and concatenate canonically.

        Per TRD §4.1:
        "The official workbook has two sheets (Year 2009-2010, Year 2010-2011) with the same
        eight columns; read all of them."
        """
        logger.info("reading online retail ii xlsx", source=str(source))
        try:
            wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
        except Exception as exc:
            raise IngestionError(
                f"Failed to open Excel workbook from {source}: {exc}",
                details={"source": str(source), "error": str(exc)},
            ) from exc

        sheet_frames: list[pd.DataFrame] = []
        try:
            for name in wb.sheetnames:
                sheet = wb[name]
                rows_iter = sheet.iter_rows(values_only=True)
                try:
                    header_row = next(rows_iter)
                except StopIteration:
                    logger.debug("skipping empty sheet", sheet=name)
                    continue

                if not header_row or all(cell is None for cell in header_row):
                    continue

                columns = [
                    str(c).strip() if c is not None else f"col_{i}"
                    for i, c in enumerate(header_row)
                ]
                data = list(rows_iter)
                if not data:
                    logger.debug("no data rows in sheet", sheet=name)
                    continue

                raw_sheet_df = pd.DataFrame(data, columns=columns)
                canonical_sheet = _process_raw_frame(
                    raw_sheet_df,
                    sheet_name=name,
                    default_store_code=default_store_code,
                )
                sheet_frames.append(canonical_sheet)
                logger.info("loaded sheet", sheet=name, rows=len(canonical_sheet))
        finally:
            wb.close()

        if not sheet_frames:
            raise IngestionError(
                f"Excel workbook {source} contained no valid data sheets",
                details={"source": str(source)},
            )

        combined = pd.concat(sheet_frames, ignore_index=True)
        logger.info("combined all sheets", total_rows=len(combined), sheets=len(sheet_frames))
        return combined

    @classmethod
    def read(
        cls,
        source: str | os.PathLike[str] | BinaryIO | TextIO,
        default_store_code: str = "ONLINE",
    ) -> pd.DataFrame:
        """Auto-detect format (CSV or XLSX) and produce the canonical DataFrame."""
        path_str = str(source).lower()
        if path_str.endswith(".xlsx") or path_str.endswith(".xls"):
            return cls.read_xlsx(
                cast(str | os.PathLike[str] | BinaryIO, source),
                default_store_code=default_store_code,
            )
        return cls.read_csv(source, default_store_code=default_store_code)
