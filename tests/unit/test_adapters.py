"""Unit tests for ingestion adapters (Online Retail II and Generic CSV)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import openpyxl
import pandas as pd
import pytest
from openpyxl.worksheet.worksheet import Worksheet

from retailpulse.errors import IngestionError
from retailpulse.ingestion.adapters.generic_csv import GenericCSVAdapter
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter
from retailpulse.ingestion.schemas import (
    CANONICAL_COLUMNS,
    canonicalize_customer_id,
    canonicalize_invoice_no,
    canonicalize_stock_code,
)


def test_canonicalize_helpers() -> None:
    """Verify customer ID, invoice number, and stock code canonicalization."""
    # Customer ID canonicalization per TRD §4.3 rule 1
    assert canonicalize_customer_id(13085) == "13085"
    assert canonicalize_customer_id(13085.0) == "13085"
    assert canonicalize_customer_id("13085.0") == "13085"
    assert canonicalize_customer_id(" 13085 ") == "13085"
    assert canonicalize_customer_id(None) is None
    assert canonicalize_customer_id("") is None
    assert canonicalize_customer_id("nan") is None

    # StockCode canonicalization
    assert canonicalize_stock_code(" 85123a ") == "85123A"
    assert canonicalize_stock_code("dcgs0076") == "DCGS0076"
    assert canonicalize_stock_code(None) == ""

    # Invoice normalization
    assert canonicalize_invoice_no(" 536365.0 ") == "536365"
    assert canonicalize_invoice_no("C536379") == "C536379"


def test_online_retail_ii_fixture_read() -> None:
    """Verify Online Retail II adapter parses tests/fixtures/mini_retail.csv correctly."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} does not exist.")

    df = OnlineRetailIIAdapter.read_csv(fixture_path)

    # Check canonical column presence
    for col in CANONICAL_COLUMNS:
        assert col in df.columns

    assert "sheet_name" in df.columns
    assert "sheet_row_idx" in df.columns

    # Verify types
    assert pd.api.types.is_string_dtype(df["invoice_no"]) or pd.api.types.is_object_dtype(
        df["invoice_no"]
    )
    assert pd.api.types.is_string_dtype(df["sku"]) or pd.api.types.is_object_dtype(df["sku"])
    assert pd.api.types.is_datetime64_any_dtype(df["invoice_ts"])
    assert pd.api.types.is_numeric_dtype(df["quantity"])
    assert pd.api.types.is_numeric_dtype(df["unit_price"])

    # Verify customer ID has no decimal .0
    non_null_cust = df["customer_id"].dropna()
    assert len(non_null_cust) > 0
    assert not any(".0" in cid for cid in non_null_cust)

    # Verify default store code
    assert (df["store_code"] == "ONLINE").all()


def test_online_retail_ii_column_variants() -> None:
    """Verify Online Retail II adapter accepts alternate column aliases."""
    # Variant 2: InvoiceNo, UnitPrice, CustomerID
    csv_data = (
        "InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
        "536365,85123A,WHITE HEART T-LIGHT,6,2010-12-01 08:26:00,2.55,17850,United Kingdom\n"
        "C536379,D,Discount,-1,2010-12-01 09:41:00,27.50,14527,United Kingdom\n"
    )
    df = OnlineRetailIIAdapter.read_csv(io.StringIO(csv_data))
    assert len(df) == 2
    assert df.iloc[0]["invoice_no"] == "536365"
    assert df.iloc[0]["sku"] == "85123A"
    assert df.iloc[0]["unit_price"] == 2.55
    assert df.iloc[0]["customer_id"] == "17850"
    assert df.iloc[1]["invoice_no"] == "C536379"

    # Variant 3: snake_case / lower-case
    csv_data_lower = (
        "invoice_no,sku,description,qty,invoice_date,unit_price,customer_id,country\n"
        "536366,22633,HAND WARMER UNION JACK,6,2010-12-01 08:28:00,1.85,17850,United Kingdom\n"
    )
    df_lower = OnlineRetailIIAdapter.read_csv(io.StringIO(csv_data_lower))
    assert len(df_lower) == 1
    assert df_lower.iloc[0]["invoice_no"] == "536366"
    assert df_lower.iloc[0]["sku"] == "22633"


def test_online_retail_ii_missing_column_fails() -> None:
    """Verify Online Retail II adapter raises IngestionError on missing required column."""
    bad_csv = "Invoice,Description,Quantity,InvoiceDate,Price,Country\n1,desc,1,2010-12-01,1.0,UK\n"
    with pytest.raises(IngestionError, match=r"missing required column.*StockCode"):
        OnlineRetailIIAdapter.read_csv(io.StringIO(bad_csv))


def test_online_retail_ii_multi_sheet_xlsx(tmp_path: Path) -> None:
    """Verify Online Retail II adapter concatenates all sheets from an Excel workbook."""
    wb = openpyxl.Workbook()
    # Sheet 1: Year 2009-2010
    ws1 = cast(Worksheet, wb.active)
    ws1.title = "Year 2009-2010"
    ws1.append(
        [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]
    )
    ws1.append(
        [
            "489434",
            "85048",
            "15CM CHRISTMAS GLASS BALL 20 LIGHTS",
            12,
            "2009-12-01 07:45:00",
            6.95,
            13085,
            "United Kingdom",
        ]
    )
    ws1.append(
        [
            "489435",
            "79323P",
            "PINK CHERRY LIGHTS",
            12,
            "2009-12-01 07:46:00",
            6.75,
            13085,
            "United Kingdom",
        ]
    )

    # Sheet 2: Year 2010-2011
    ws2 = wb.create_sheet(title="Year 2010-2011")
    ws2.append(
        [
            "Invoice",
            "StockCode",
            "Description",
            "Quantity",
            "InvoiceDate",
            "Price",
            "Customer ID",
            "Country",
        ]
    )
    ws2.append(
        [
            "536365",
            "85123A",
            "WHITE HANGING HEART T-LIGHT HOLDER",
            6,
            "2010-12-01 08:26:00",
            2.55,
            17850,
            "United Kingdom",
        ]
    )
    ws2.append(
        [
            "536366",
            "22633",
            "HAND WARMER UNION JACK",
            6,
            "2010-12-01 08:28:00",
            1.85,
            17850,
            "United Kingdom",
        ]
    )

    wb_path = tmp_path / "test_retail.xlsx"
    wb.save(wb_path)

    # Read with adapter
    df = OnlineRetailIIAdapter.read_xlsx(wb_path)
    assert len(df) == 4

    # Verify both sheets are present with provenance
    assert (df["sheet_name"] == "Year 2009-2010").sum() == 2
    assert (df["sheet_name"] == "Year 2010-2011").sum() == 2
    assert list(df["sheet_row_idx"]) == [0, 1, 0, 1]

    # Verify values
    assert df.iloc[0]["invoice_no"] == "489434"
    assert df.iloc[2]["invoice_no"] == "536365"
    assert df.iloc[0]["customer_id"] == "13085"


def test_generic_csv_adapter() -> None:
    """Verify GenericCSVAdapter reads valid generic CSV and applies defaults."""
    csv_data = (
        "invoice_no,invoice_date,sku,quantity,unit_price,customer_id\n"
        "INV-1001,2023-01-15T10:30:00,SKU-A,5,19.99,CUST-99\n"
        "INV-1002,2023-01-15T11:00:00,SKU-B,1,49.50,\n"
    )
    df = GenericCSVAdapter.read_csv(io.StringIO(csv_data))
    assert len(df) == 2
    assert df.iloc[0]["invoice_no"] == "INV-1001"
    assert df.iloc[0]["sku"] == "SKU-A"
    assert df.iloc[0]["unit_price"] == 19.99
    assert df.iloc[0]["customer_id"] == "CUST-99"
    assert df.iloc[0]["store_code"] == "ONLINE"
    assert df.iloc[0]["country"] == "GB"

    # Second row has null customer
    assert pd.isna(df.iloc[1]["customer_id"])


def test_generic_csv_missing_column() -> None:
    """Verify GenericCSVAdapter raises IngestionError when required column is missing."""
    csv_data = "invoice_no,sku,quantity\nINV-1,SKU-1,10\n"
    with pytest.raises(IngestionError, match="missing required column"):
        GenericCSVAdapter.read_csv(io.StringIO(csv_data))
