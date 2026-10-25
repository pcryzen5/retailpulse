"""Unit tests for data quality schemas, cleaning rules, batch gates, and outlier detection.

Verifies each rule in TRD §4.3:
  - Rule 1: Type coercion and quarantine checks
  - Rule 2(a): Cross-sheet overlap duplicate detection
  - Rule 2(b): Exact duplicate rows
  - Rule 3: Returns, invalid cancellations (C with qty > 0), and stock adjustments
  - Rule 4: Non-product SKUs (config-driven; DCGS* real products preserved)
  - Rule 5: Non-positive unit prices
  - Rule 6: Customer ID presence accounting
  - Rule 7: Source row hash uniqueness
  - Rule 8: Batch gate (MAX_REJECT_RATIO = 0.20)
  - Rule 9: Outlier flag (OUTLIER_QTY = 50,000)
Plus Pandera schema validation tests and fixture integration.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retailpulse.errors import DataQualityError
from retailpulse.etl.quality import (
    OUTLIER_QTY,
    check_cross_sheet_overlap,
    check_customer_id_distribution,
    check_exact_duplicates,
    check_hash_uniqueness,
    check_non_positive_price,
    check_non_product_skus,
    check_type_coercion,
    classify_returns,
    cleaned_sales_line_schema,
    enforce_batch_gate,
    evaluate_batch_gate,
    flag_outliers,
    identify_invalid_cancellations,
    identify_stock_adjustments,
    load_non_product_skus,
    raw_sales_line_schema,
    run_data_quality_checks,
    validate_returns_and_cancellations,
    validate_schema,
)
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter


def test_rule_1_type_coercion_and_quarantine() -> None:
    """Verify TRD §4.3 Rule 1: type coercion catches malformed dates, prices, empty fields."""
    sample_df = pd.DataFrame(
        {
            "invoice_no": ["536365", "536366", "", "536368", "536369"],
            "invoice_ts": [
                pd.Timestamp("2010-12-01 08:26:00"),
                pd.NaT,  # Unparseable date
                pd.Timestamp("2010-12-01 08:30:00"),
                pd.Timestamp("2010-12-01 08:34:00"),
                pd.Timestamp("2010-12-01 08:35:00"),
            ],
            "sku": ["85123A", "71053", "22633", "", "22720"],  # row 3 has empty sku
            "description": ["WHITE HEART", "LANTERN", "WARMER", "WARMER", "CAKE TINS"],
            "quantity": [6.0, 10.0, 2.0, float("nan"), 4.0],  # row 3 has NaN qty
            "unit_price": [2.55, 3.39, float("nan"), 1.85, 4.95],  # row 2 has NaN price
            "customer_id": ["17850", "17850", "17850", "17850", None],
            "country": ["United Kingdom"] * 5,
            "store_code": ["ONLINE"] * 5,
        }
    )

    failed_mask, finding = check_type_coercion(sample_df)

    # Rows 1 (NaT date), 2 (empty invoice & NaN price), 3 (empty sku & NaN qty) fail
    assert bool(failed_mask.iloc[0]) is False  # Valid
    assert bool(failed_mask.iloc[1]) is True  # Bad date
    assert bool(failed_mask.iloc[2]) is True  # Bad price & empty invoice
    assert bool(failed_mask.iloc[3]) is True  # Bad qty & empty sku
    assert bool(failed_mask.iloc[4]) is False  # Valid (null customer_id is allowed)

    assert finding.check_name == "type_coercion"
    assert finding.affected_rows == 3
    assert finding.passed is False
    assert finding.severity == "warning"
    assert finding.details["invalid_dates"] == 1
    assert finding.details["invalid_quantities"] == 1
    assert finding.details["invalid_prices"] == 1
    assert finding.details["empty_invoices"] == 1
    assert finding.details["empty_skus"] == 1


def test_rule_2a_cross_sheet_overlap_duplicates() -> None:
    """Verify TRD §4.3 Rule 2(a): cross-sheet duplicates dropped; within-sheet repeats kept."""
    # Sheet 1 has row A twice (legitimate repeat within sheet)
    # Sheet 2 has row A twice (overlapping copy of sheet 1)
    base_row = {
        "invoice_no": "536365",
        "invoice_ts": pd.Timestamp("2010-12-01 08:26:00"),
        "sku": "85123A",
        "description": "WHITE HEART",
        "quantity": 6.0,
        "unit_price": 2.55,
        "customer_id": "17850",
        "country": "United Kingdom",
        "store_code": "ONLINE",
    }
    different_row = {
        "invoice_no": "536366",
        "invoice_ts": pd.Timestamp("2010-12-01 08:28:00"),
        "sku": "71053",
        "description": "LANTERN",
        "quantity": 2.0,
        "unit_price": 3.39,
        "customer_id": "17850",
        "country": "United Kingdom",
        "store_code": "ONLINE",
    }

    df = pd.DataFrame(
        [
            dict(base_row, sheet_name="Year 2009-2010"),  # Row A occ 0 in Sheet 1 (keep)
            dict(base_row, sheet_name="Year 2009-2010"),  # Row A occ 1 in Sheet 1 (keep)
            dict(different_row, sheet_name="Year 2009-2010"),
            dict(base_row, sheet_name="Year 2010-2011"),  # Row A occ 0 in Sheet 2 (DROP overlap)
            dict(base_row, sheet_name="Year 2010-2011"),  # Row A occ 1 in Sheet 2 (DROP overlap)
            dict(different_row, sheet_name="Year 2010-2011"),  # Sheet 2 (DROP overlap)
        ]
    )

    is_cross_sheet_dup, finding = check_cross_sheet_overlap(df)

    assert finding.check_name == "cross_sheet_overlap"
    assert finding.affected_rows == 3
    # Rows 0, 1, 2 should not be duplicate
    assert not is_cross_sheet_dup.iloc[0]
    assert not is_cross_sheet_dup.iloc[1]
    assert not is_cross_sheet_dup.iloc[2]
    # Rows 3, 4, 5 in second sheet should be flagged as cross-sheet duplicates
    assert is_cross_sheet_dup.iloc[3]
    assert is_cross_sheet_dup.iloc[4]
    assert is_cross_sheet_dup.iloc[5]


def test_rule_2b_exact_duplicate_rows() -> None:
    """Verify TRD §4.3 Rule 2(b): exact duplicate rows within dataset are flagged."""
    row = {
        "invoice_no": "538000",
        "invoice_ts": pd.Timestamp("2010-12-08 15:30:00"),
        "sku": "22423",
        "description": "REGENCY CAKESTAND",
        "quantity": 1.0,
        "unit_price": 12.75,
        "customer_id": "14000",
        "country": "United Kingdom",
        "store_code": "ONLINE",
    }
    df = pd.DataFrame([row, row, row])

    is_dup, finding = check_exact_duplicates(df)

    assert finding.check_name == "exact_duplicates"
    assert finding.affected_rows == 2
    assert not is_dup.iloc[0]
    assert is_dup.iloc[1]
    assert is_dup.iloc[2]


def test_rule_3_returns_and_cancellations() -> None:
    """Verify TRD §4.3 Rule 3: return polarity, invalid C-invoices, and stock adjustments."""
    df = pd.DataFrame(
        {
            "invoice_no": [
                "536365",  # Normal sale
                "C536379",  # Valid return: C invoice, negative qty
                "C539999",  # INVALID CANCELLATION: C invoice with positive quantity!
                "536380",  # Negative qty, with customer & price > 0 -> return
                "536381",  # Negative qty, null customer -> stock adjustment
                "536382",  # Negative qty, zero price -> stock adjustment
            ],
            "quantity": [6, -1, 2, -3, -5, -10],
            "unit_price": [2.55, 27.50, 7.65, 4.95, 0.00, 0.00],
            "customer_id": ["17850", "14527", "12347", "13085", None, "14000"],
        }
    )

    # 1. Invalid cancellations
    invalid_canc = identify_invalid_cancellations(df)
    assert invalid_canc.tolist() == [False, False, True, False, False, False]

    # 2. Stock adjustments
    stock_adj = identify_stock_adjustments(df)
    assert stock_adj.tolist() == [False, False, False, False, True, True]

    # 3. Classify valid returns
    is_return = classify_returns(df)
    # Row 1 (C536379, -1) and Row 3 (536380, -3 with customer & price) are valid returns
    assert is_return.tolist() == [False, True, False, True, False, False]

    # 4. validate_returns_and_cancellations helper
    _, _, f_inv, f_adj = validate_returns_and_cancellations(df)
    assert f_inv.check_name == "invalid_cancellation_qty"
    assert f_inv.affected_rows == 1
    assert f_inv.passed is False
    assert f_inv.severity == "error"

    assert f_adj.check_name == "stock_adjustments"
    assert f_adj.affected_rows == 2
    assert f_adj.passed is True


def test_rule_4_non_product_skus() -> None:
    """Verify TRD §4.3 Rule 4: non-product SKUs flagged from config; DCGS* kept as products."""
    skus = load_non_product_skus()
    assert "POST" in skus
    assert "DOT" in skus
    assert "M" in skus
    assert "BANK CHARGES" in skus
    assert "CRUK" in skus

    # Verify DCGS codes are NEVER treated as non-product
    assert not any(sku.startswith("DCGS") for sku in skus)

    test_df = pd.DataFrame(
        {
            "sku": [
                "85123A",  # Real product
                "POST",  # Non-product
                "dot",  # Non-product (case variant)
                "DCGS0003",  # Real product with DCGS prefix!
                "DCGSSGIRL",  # Real product with DCGS prefix!
                "M",  # Non-product
                "BANK CHARGES",  # Non-product
            ]
        }
    )

    is_non_prod, finding = check_non_product_skus(test_df, non_product_skus=skus)
    # POST, dot, M, BANK CHARGES are non-product (4 rows)
    assert is_non_prod.tolist() == [False, True, True, False, False, True, True]
    assert finding.affected_rows == 4

    def _check_dcgs(sku_list: list[str]) -> None:
        for s in sku_list:
            if s.startswith("DCGS"):
                raise DataQualityError(f"DCGS code '{s}' found")

    bad_sku_list = ["POST", "DCGS0001"]
    with pytest.raises(DataQualityError, match="DCGS code 'DCGS0001' found"):
        _check_dcgs(bad_sku_list)


def test_rule_5_non_positive_price() -> None:
    """Verify TRD §4.3 Rule 5: lines with unit_price <= 0 are detected."""
    df = pd.DataFrame({"unit_price": [2.55, 0.0, -1.5, 10.0, float("nan")]})
    is_non_pos, finding = check_non_positive_price(df)

    assert is_non_pos.tolist() == [False, True, True, False, True]
    assert finding.affected_rows == 3
    assert finding.passed is False
    assert finding.severity == "warning"


def test_rule_6_customer_id_distribution() -> None:
    """Verify TRD §4.3 Rule 6: customer ID missingness and distribution accounting."""
    df = pd.DataFrame(
        {
            "customer_id": ["12345", None, "12346", "", "nan", "12347"],
        }
    )
    # Empty string and 'nan' or None are considered missing
    df_clean = df.copy()
    df_clean["customer_id"] = df_clean["customer_id"].replace({"": None, "nan": None})

    finding = check_customer_id_distribution(df_clean)
    assert finding.check_name == "customer_id_presence"
    assert finding.affected_rows == 3
    assert finding.details["total"] == 6
    assert finding.details["missing_ratio"] == 0.5


def test_rule_7_source_row_hash_uniqueness() -> None:
    """Verify TRD §4.3 Rule 7: source row hash uniqueness check."""
    hashes = pd.Series(["a" * 64, "b" * 64, "c" * 64])
    finding = check_hash_uniqueness(hashes)
    assert finding.passed is True
    assert finding.affected_rows == 0

    dup_hashes = pd.Series(["a" * 64, "b" * 64, "a" * 64])
    finding_dup = check_hash_uniqueness(dup_hashes)
    assert finding_dup.passed is False
    assert finding_dup.affected_rows == 1
    assert finding_dup.severity == "error"


def test_rule_8_batch_gate_max_reject_ratio() -> None:
    """Verify TRD §4.3 Rule 8: batch gate rejects when > 20% rows fail validation."""
    # 10% rejected -> passes
    res, finding = evaluate_batch_gate(total_rows=1000, rejected_rows=100)
    assert res.passed is True
    assert res.reject_ratio == 0.10
    assert finding.passed is True
    assert finding.severity == "info"

    # Exactly 20% rejected -> passes (<= MAX_REJECT_RATIO)
    res_20, finding_20 = evaluate_batch_gate(total_rows=1000, rejected_rows=200)
    assert res_20.passed is True
    assert finding_20.passed is True

    # 21% rejected -> FAILS batch gate
    res_fail, finding_fail = evaluate_batch_gate(total_rows=1000, rejected_rows=210)
    assert res_fail.passed is False
    assert res_fail.reject_ratio == 0.21
    assert finding_fail.passed is False
    assert finding_fail.severity == "error"

    # Enforce raises DataQualityError
    with pytest.raises(DataQualityError, match=r"Batch gate rejected.*21\.00%"):
        enforce_batch_gate(total_rows=1000, rejected_rows=210)


def test_rule_9_outlier_flagging() -> None:
    """Verify TRD §4.3 Rule 9: lines with |quantity| >= 50,000 are flagged as outliers."""
    df = pd.DataFrame(
        {
            "invoice_no": ["541431", "C541433", "536365"],
            "sku": ["23166", "23166", "85123A"],
            "quantity": [74215, -74215, 12],
        }
    )

    is_outlier, finding = flag_outliers(df, outlier_qty=OUTLIER_QTY)

    assert is_outlier.tolist() == [True, True, False]
    assert finding.check_name == "outlier_quantity"
    assert finding.affected_rows == 2
    assert finding.passed is True
    assert finding.severity == "warning"
    assert len(finding.details["outliers"]) == 2


def test_pandera_raw_sales_line_schema() -> None:
    """Verify Pandera raw_sales_line_schema validates compliant canonical frames."""
    valid_df = pd.DataFrame(
        {
            "invoice_no": ["536365", "536366"],
            "invoice_ts": [
                pd.Timestamp("2010-12-01 08:26:00"),
                pd.Timestamp("2010-12-01 08:28:00"),
            ],
            "sku": ["85123A", "71053"],
            "description": ["WHITE HEART", "LANTERN"],
            "quantity": [6.0, -1.0],
            "unit_price": [2.55, 3.39],
            "customer_id": ["17850", None],
            "country": ["United Kingdom", "France"],
            "store_code": ["ONLINE", "ONLINE"],
        }
    )
    passed, errors = validate_schema(valid_df, raw_sales_line_schema)
    assert passed is True
    assert errors == []

    # Frame with empty invoice_no should fail check
    invalid_df = valid_df.copy()
    invalid_df["invoice_no"] = ["", "536366"]
    passed_inv, errors_inv = validate_schema(invalid_df, raw_sales_line_schema)
    assert passed_inv is False
    assert len(errors_inv) > 0


def test_pandera_cleaned_sales_line_schema() -> None:
    """Verify Pandera cleaned_sales_line_schema enforces business invariants."""
    valid_cleaned = pd.DataFrame(
        {
            "invoice_no": ["536365", "C536379"],
            "invoice_ts": [
                pd.Timestamp("2010-12-01 08:26:00"),
                pd.Timestamp("2010-12-01 09:41:00"),
            ],
            "sku": ["85123A", "22633"],
            "description": ["WHITE HEART", "WARMER"],
            "quantity": [6, -1],
            "unit_price": [2.55, 1.85],
            "customer_key": ["a" * 64, None],
            "country": ["United Kingdom", "United Kingdom"],
            "store_code": ["ONLINE", "ONLINE"],
            "is_return": [False, True],
            "is_product": [True, True],
            "is_outlier": [False, False],
            "source_row_hash": ["1" * 64, "2" * 64],
        }
    )
    passed, errors = validate_schema(valid_cleaned, cleaned_sales_line_schema)
    assert passed is True
    assert errors == []

    # Polarity violation: return with positive quantity
    invalid_polarity = valid_cleaned.copy()
    invalid_polarity["is_return"] = [True, True]  # row 0 has positive qty (6) but is_return=True!
    passed_pol, errors_pol = validate_schema(invalid_polarity, cleaned_sales_line_schema)
    assert passed_pol is False
    assert any("return_quantity_polarity_match" in str(e) for e in errors_pol)


def test_quality_checks_on_fixture() -> None:
    """Verify run_data_quality_checks executes across tests/fixtures/mini_retail.csv."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    if not fixture_path.exists():
        pytest.skip("mini_retail.csv fixture not found")

    df = OnlineRetailIIAdapter.read_csv(fixture_path)
    findings = run_data_quality_checks(df)

    # Check that findings cover the critical rules
    finding_map = {f.check_name: f for f in findings}
    assert "type_coercion" in finding_map
    assert "invalid_cancellation_qty" in finding_map
    assert "stock_adjustments" in finding_map
    assert "non_product_skus" in finding_map
    assert "outlier_quantity" in finding_map

    # mini_retail.csv contains:
    # - 1 invalid cancellation (C539999 with quantity 2)
    assert finding_map["invalid_cancellation_qty"].affected_rows == 1
    assert finding_map["invalid_cancellation_qty"].passed is False

    # - 2 outlier lines (74215 and -74215)
    assert finding_map["outlier_quantity"].affected_rows == 2

    # - Non-product SKUs (POST, DOT, M, BANK CHARGES, C2)
    assert finding_map["non_product_skus"].affected_rows > 0

    # - Bad coercion rows (INVALID-DATE-TIME, FREE price)
    assert finding_map["type_coercion"].affected_rows == 2
