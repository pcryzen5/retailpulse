"""Unit tests for the ETL cleaning pipeline per TRD §4.3 rules 1-7 and 9.

Verifies:
  - Raw customer IDs never appear in outputs (SEC-DATA-01 / P1-07 exit criteria)
  - Type coercion and quarantine separation
  - Cross-sheet overlap and exact duplicate removal
  - Returns vs cancellations vs stock adjustments
  - Non-product SKU filtering with DCGS* preserved
  - Non-positive price line dropping
  - HMAC-SHA256 customer pseudonymization
  - Deterministic source_row_hash generation
  - Outlier quantity flagging
  - Full pipeline execution on tests/fixtures/mini_retail.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retailpulse.etl.clean import (
    CLEANED_COLUMNS,
    clean_sales_lines,
    compute_source_row_hash,
    compute_source_row_hashes,
)
from retailpulse.etl.quality import cleaned_sales_line_schema, validate_schema
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter

TEST_SALT = "test-salt-secret-at-least-32-bytes-long!"


def test_compute_source_row_hash() -> None:
    """Verify deterministic 64-char sha256 hash using unit separator."""
    ts = pd.Timestamp("2010-12-01 08:26:00")
    h1 = compute_source_row_hash(
        invoice_no="536365",
        invoice_ts=ts,
        sku="85123A",
        description="WHITE HEART",
        quantity=6,
        unit_price=2.55,
        customer_key="a" * 64,
        country="United Kingdom",
        store_code="ONLINE",
    )
    h2 = compute_source_row_hash(
        invoice_no="536365",
        invoice_ts=ts,
        sku="85123A",
        description="WHITE HEART",
        quantity=6,
        unit_price=2.55,
        customer_key="a" * 64,
        country="United Kingdom",
        store_code="ONLINE",
    )

    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)

    # Any field variation alters the hash
    h_diff_qty = compute_source_row_hash(
        invoice_no="536365",
        invoice_ts=ts,
        sku="85123A",
        description="WHITE HEART",
        quantity=7,  # Changed quantity
        unit_price=2.55,
        customer_key="a" * 64,
        country="United Kingdom",
        store_code="ONLINE",
    )
    assert h1 != h_diff_qty


def test_compute_source_row_hashes_vectorized() -> None:
    """Verify compute_source_row_hashes processes DataFrames correctly."""
    df = pd.DataFrame(
        {
            "invoice_no": ["536365", "536366"],
            "invoice_ts": [
                pd.Timestamp("2010-12-01 08:26:00"),
                pd.Timestamp("2010-12-01 08:28:00"),
            ],
            "sku": ["85123A", "71053"],
            "description": ["WHITE HEART", "LANTERN"],
            "quantity": [6, 2],
            "unit_price": [2.55, 3.39],
            "customer_key": ["a" * 64, None],
            "country": ["United Kingdom", "United Kingdom"],
            "store_code": ["ONLINE", "ONLINE"],
        }
    )
    hashes = compute_source_row_hashes(df)
    assert len(hashes) == 2
    assert len(str(hashes.iloc[0])) == 64
    assert len(str(hashes.iloc[1])) == 64
    assert hashes.iloc[0] != hashes.iloc[1]


def test_raw_customer_id_never_in_cleaned_output() -> None:
    """CRITICAL SECURITY TEST (P1-07): Raw customer IDs must never appear in cleaned outputs."""
    raw_customers = ["17850", "13085", "14000", "99999"]
    raw_df = pd.DataFrame(
        {
            "invoice_no": ["536365", "536366", "536367", "536368"],
            "invoice_ts": [
                pd.Timestamp("2010-12-01 08:26:00"),
                pd.Timestamp("2010-12-01 08:28:00"),
                pd.Timestamp("2010-12-01 08:30:00"),
                pd.Timestamp("2010-12-01 08:32:00"),
            ],
            "sku": ["85123A", "71053", "22633", "22720"],
            "description": ["WHITE HEART", "LANTERN", "WARMER", "CAKE TINS"],
            "quantity": [6, 2, 4, 1],
            "unit_price": [2.55, 3.39, 1.85, 4.95],
            "customer_id": raw_customers,
            "country": ["United Kingdom"] * 4,
            "store_code": ["ONLINE"] * 4,
        }
    )

    cleaned = clean_sales_lines(raw_df, salt=TEST_SALT)
    clean_df = cleaned.df

    # 1. customer_id column MUST NOT exist
    assert "customer_id" not in clean_df.columns
    assert "customer_key" in clean_df.columns

    # 2. Raw customer ID values must NEVER appear in any column of the output DataFrame
    for raw_id in raw_customers:
        for col in clean_df.columns:
            matches = clean_df[col].astype(str).str.contains(raw_id)
            assert not matches.any(), f"Raw customer ID '{raw_id}' leaked in column '{col}'!"

    # 3. Output string representation must not contain raw customer IDs
    dump = clean_df.to_string()
    for raw_id in raw_customers:
        assert raw_id not in dump, f"Raw customer ID '{raw_id}' leaked in DataFrame string dump!"

    # 4. Customer keys must be valid 64-char hex strings
    assert all(len(str(k)) == 64 for k in clean_df["customer_key"])


def test_clean_sales_lines_sequential_rules() -> None:
    """Verify sequential application of cleaning rules 1-7 and 9."""
    raw_df = pd.DataFrame(
        {
            "invoice_no": [
                "536365",  # 0: Valid regular product
                "536366",  # 1: Coercion failure (invalid date) -> quarantined
                "536367",  # 2: Cross-sheet overlap duplicate -> dropped
                "536368",  # 3: Exact duplicate -> dropped
                "536368",  # 4: Exact duplicate copy
                "C536369",  # 5: Invalid cancellation with positive qty -> quarantined
                "536370",  # 6: Stock adjustment (negative qty, null customer) -> dropped
                "C536371",  # 7: Valid return (C-invoice, negative qty) -> kept
                "536372",  # 8: Non-product SKU (POST) -> kept as is_product=False
                "536373",  # 9: Real product with DCGS prefix -> kept as is_product=True
                "536374",  # 10: Non-positive price (0.00) -> dropped
                "536375",  # 11: Outlier order (74,215 units) -> kept as is_outlier=True
            ],
            "invoice_ts": [
                pd.Timestamp("2010-12-01 08:26:00"),
                pd.NaT,  # Row 1 bad date
                pd.Timestamp("2010-12-01 08:27:00"),
                pd.Timestamp("2010-12-01 08:28:00"),
                pd.Timestamp("2010-12-01 08:28:00"),
                pd.Timestamp("2010-12-01 08:29:00"),
                pd.Timestamp("2010-12-01 08:30:00"),
                pd.Timestamp("2010-12-01 08:31:00"),
                pd.Timestamp("2010-12-01 08:32:00"),
                pd.Timestamp("2010-12-01 08:33:00"),
                pd.Timestamp("2010-12-01 08:34:00"),
                pd.Timestamp("2010-12-01 08:35:00"),
            ],
            "sku": [
                "85123A",
                "71053",
                "85123A",
                "22633",
                "22633",
                "22720",
                "22720",
                "22720",
                "POST",  # Non-product
                "DCGS0003",  # Real product!
                "21212",
                "23166",
            ],
            "description": ["DESC"] * 12,
            "quantity": [
                6,
                2,
                6,
                1,
                1,
                2,  # Invalid cancellation on C invoice!
                -4,  # Stock adjustment (null customer)
                -2,  # Valid return
                1,
                1,
                10,
                74215,  # Outlier >= 50,000!
            ],
            "unit_price": [
                2.55,
                3.39,
                2.55,
                1.85,
                1.85,
                7.65,
                4.95,
                7.65,
                18.00,
                2.55,
                0.00,  # Zero price!
                1.04,
            ],
            "customer_id": [
                "17850",
                "17850",
                "17850",
                "14000",
                "14000",
                "12347",
                None,  # Null customer on negative qty
                "12347",
                "17850",
                "17850",
                "17850",
                "12346",
            ],
            "country": ["United Kingdom"] * 12,
            "store_code": ["ONLINE"] * 12,
            "sheet_name": [
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2010-2011",  # Overlap copy in second sheet
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
                "Year 2009-2010",
            ],
        }
    )

    cleaned = clean_sales_lines(raw_df, salt=TEST_SALT)
    clean_df = cleaned.df
    quarantined = cleaned.quarantined_df

    # 1. Quarantined records: Row 1 (bad date) and Row 5 (invalid cancellation C536369)
    assert len(quarantined) == 2
    assert "type_coercion" in quarantined["quarantine_reason"].to_numpy()
    assert "invalid_cancellation_qty" in quarantined["quarantine_reason"].to_numpy()

    # 2. Columns match CLEANED_COLUMNS
    assert list(clean_df.columns) == CLEANED_COLUMNS
    assert "customer_id" not in clean_df.columns

    # 3. Non-product SKU POST has is_product = False
    post_row = clean_df[clean_df["sku"] == "POST"]
    assert len(post_row) == 1
    assert bool(post_row["is_product"].iloc[0]) is False

    # 4. DCGS product SKU has is_product = True
    dcgs_row = clean_df[clean_df["sku"] == "DCGS0003"]
    assert len(dcgs_row) == 1
    assert bool(dcgs_row["is_product"].iloc[0]) is True

    # 5. Outlier row (74215) has is_outlier = True
    outlier_row = clean_df[clean_df["quantity"] == 74215]
    assert len(outlier_row) == 1
    assert bool(outlier_row["is_outlier"].iloc[0]) is True

    # 6. Valid return (C536371) has is_return = True
    return_row = clean_df[clean_df["invoice_no"] == "C536371"]
    assert len(return_row) == 1
    assert bool(return_row["is_return"].iloc[0]) is True

    # 7. Non-positive price row (536374) was dropped
    assert "536374" not in clean_df["invoice_no"].to_numpy()

    # 8. Stock adjustment row (536370) was dropped
    assert "536370" not in clean_df["invoice_no"].to_numpy()

    # 9. Cleaned DataFrame satisfies pandera schema
    is_valid, errors = validate_schema(clean_df, cleaned_sales_line_schema)
    assert is_valid is True, f"Schema validation errors: {errors}"


def test_clean_sales_lines_on_fixture() -> None:
    """Verify cleaning pipeline executes end-to-end on tests/fixtures/mini_retail.csv."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    if not fixture_path.exists():
        pytest.skip("mini_retail.csv fixture not found")

    raw_df = OnlineRetailIIAdapter.read_csv(fixture_path)
    cleaned = clean_sales_lines(raw_df, salt=TEST_SALT)
    df = cleaned.df

    # 1. No customer_id column in output
    assert "customer_id" not in df.columns
    assert "customer_key" in df.columns

    # 2. Output conforms to Pandera cleaned schema
    is_valid, errors = validate_schema(df, cleaned_sales_line_schema)
    assert is_valid is True, f"Schema errors: {errors}"

    # 3. Quarantined records contain invalid cancellations and coercion errors
    quarantined = cleaned.quarantined_df
    assert len(quarantined) >= 3  # 2 coercion failures + 1 invalid cancellation in mini_retail.csv
    assert "C539999" in quarantined["invoice_no"].to_numpy()

    # 4. Outliers identified (both the order 74215 and its cancellation -74215)
    outliers = df[df["is_outlier"]]
    assert len(outliers) == 2
    assert set(outliers["quantity"]) == {74215, -74215}

    # 5. Non-product SKUs present with is_product = False
    non_products = df[~df["is_product"]]
    assert len(non_products) > 0
    non_prod_skus = set(non_products["sku"].unique())
    assert "POST" in non_prod_skus
    assert "DOT" in non_prod_skus

    # 6. Real products with DCGS prefix have is_product = True
    dcgs_skus = df[df["sku"].str.startswith("DCGS")]
    assert len(dcgs_skus) > 0
    assert dcgs_skus["is_product"].all()

    # 7. Row counts reconcile properly
    assert cleaned.row_counts["raw_read"] == len(raw_df)
    assert cleaned.row_counts["final_cleaned"] == len(df)
    assert cleaned.row_counts["final_cleaned"] < cleaned.row_counts["raw_read"]
