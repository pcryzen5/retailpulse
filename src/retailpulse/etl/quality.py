"""Data quality schemas, validation rules, batch gates, and outlier detection.

Implements TRD §4.3 cleaning & quality rules:
- Pandera schema definitions for canonical raw and cleaned frames
- Batch gate checking against MAX_REJECT_RATIO (0.20)
- Outlier quantity flagging against OUTLIER_QTY (50,000)
- Cancellation & return validation (rejection of C-invoices with quantity > 0)
- Non-product SKU filtering via explicit configuration
- Comprehensive quality check suite generating findings for ops.data_quality_result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pandera.pandas as pa
import yaml

from retailpulse.errors import DataQualityError
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Constants per TRD §4.3
MAX_REJECT_RATIO: float = 0.20  # Batch gate: mark rejected if >20% rows rejected
OUTLIER_QTY: int = 50_000  # Lines with |qty| >= 50,000 marked is_outlier = true

# Severity levels matching ops.data_quality_result check constraint
SeverityType = Literal["info", "warning", "error"]

# Fallback explicit non-product SKUs per TRD §4.3 rule 4
DEFAULT_NON_PRODUCT_SKUS: frozenset[str] = frozenset(
    {
        "POST",
        "DOT",
        "M",
        "C2",
        "D",
        "S",
        "BANK CHARGES",
        "ADJUST",
        "AMAZONFEE",
        "PADS",
        "CRUK",
        "TEST001",
        "GIFT_0001_10",
        "GIFT_0001_20",
        "GIFT_0001_30",
        "GIFT_0001_40",
        "GIFT_0001_50",
    }
)


@dataclass(frozen=True)
class QualityFinding:
    """Finding for a single quality rule check, corresponding to ops.data_quality_result."""

    check_name: str
    severity: SeverityType
    passed: bool
    affected_rows: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchGateResult:
    """Summary of batch gate evaluation against MAX_REJECT_RATIO."""

    passed: bool
    total_rows: int
    rejected_rows: int
    reject_ratio: float
    max_allowed_ratio: float = MAX_REJECT_RATIO


# Raw canonical frame schema before ETL cleaning (TRD §4.2)
raw_sales_line_schema = pa.DataFrameSchema(
    columns={
        "invoice_no": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check(lambda s: s.str.strip().str.len() > 0, name="non_empty_invoice_no"),
            description="Normalized invoice number as string",
        ),
        "invoice_ts": pa.Column(
            pa.DateTime,
            nullable=False,
            description="Parsed invoice timestamp",
        ),
        "sku": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check(lambda s: s.str.strip().str.len() > 0, name="non_empty_sku"),
            description="Normalized stock code / SKU",
        ),
        "description": pa.Column(
            pa.String,
            nullable=True,
            description="Product description",
        ),
        "quantity": pa.Column(
            pa.Float,
            nullable=False,
            description="Transaction quantity",
        ),
        "unit_price": pa.Column(
            pa.Float,
            nullable=False,
            description="Unit price in local currency",
        ),
        "customer_id": pa.Column(
            pa.String,
            nullable=True,
            description="Canonical customer ID as integer text or None",
        ),
        "country": pa.Column(
            pa.String,
            nullable=False,
            description="Country of transaction",
        ),
        "store_code": pa.Column(
            pa.String,
            nullable=False,
            description="Store or channel identifier",
        ),
    },
    strict=False,
    coerce=False,
)

# Cleaned sales line schema after transformations ready for fact_sales_line
cleaned_sales_line_schema = pa.DataFrameSchema(
    columns={
        "invoice_no": pa.Column(pa.String, nullable=False),
        "invoice_ts": pa.Column(pa.DateTime, nullable=False),
        "sku": pa.Column(pa.String, nullable=False),
        "description": pa.Column(pa.String, nullable=True),
        "quantity": pa.Column(
            pa.Int,
            nullable=False,
            checks=pa.Check(lambda s: s != 0, name="non_zero_quantity"),
        ),
        "unit_price": pa.Column(
            pa.Float,
            nullable=False,
            checks=pa.Check(lambda s: s > 0, name="positive_unit_price"),
        ),
        "customer_key": pa.Column(
            pa.String,
            nullable=True,
            description="Pseudonymized customer HMAC-SHA256 hex key or None",
        ),
        "country": pa.Column(pa.String, nullable=False),
        "store_code": pa.Column(pa.String, nullable=False),
        "is_return": pa.Column(pa.Bool, nullable=False),
        "is_product": pa.Column(pa.Bool, nullable=False),
        "is_outlier": pa.Column(pa.Bool, nullable=False),
        "source_row_hash": pa.Column(
            pa.String,
            nullable=False,
            checks=pa.Check(lambda s: s.str.len() == 64, name="valid_sha256_hash"),
        ),
    },
    checks=[
        pa.Check(
            lambda df: (
                (df["is_return"] & (df["quantity"] < 0)) | (~df["is_return"] & (df["quantity"] > 0))
            ).all(),
            name="return_quantity_polarity_match",
        )
    ],
    strict=False,
    coerce=False,
)


def load_non_product_skus(config_path: Path | str | None = None) -> set[str]:
    """Load explicit non-product SKU list from configuration file.

    Per TRD §4.3 rule 4:
    "Mark non-product SKUs (is_product = false) from an explicit list in
    config/non_product_skus.yaml, never by pattern. DCGS* codes are real products: keep them."
    """
    path = Path(config_path) if config_path else Path("config/non_product_skus.yaml")
    if not path.exists():
        logger.warning("non_product_skus config file not found, using defaults", path=str(path))
        return set(DEFAULT_NON_PRODUCT_SKUS)

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        skus = data.get("non_product_skus", [])
        loaded = {str(sku).strip().upper() for sku in skus if sku}
    except Exception as exc:
        logger.error("failed to load non_product_skus yaml", error=str(exc))
        return set(DEFAULT_NON_PRODUCT_SKUS)

    # Sanity check: Ensure DCGS* codes are NEVER in the non-product list
    for sku in loaded:
        if sku.startswith("DCGS"):
            raise DataQualityError(
                f"Configuration error: DCGS code '{sku}' found in non_product_skus list. "
                "DCGS* codes are real products and must never be classified as non-product."
            )

    return loaded


def check_type_coercion(df: pd.DataFrame) -> tuple[pd.Series, QualityFinding]:
    """Identify rows that fail basic type coercion per TRD §4.3 Rule 1.

    Checks:
    - Null / unparseable invoice_ts
    - Null / unparseable quantity
    - Null / unparseable unit_price
    - Empty invoice_no or sku

    Returns:
        tuple[pd.Series, QualityFinding]: (boolean mask of invalid rows, finding)
    """
    if df.empty:
        empty_mask = pd.Series([], dtype=bool)
        return empty_mask, QualityFinding(
            check_name="type_coercion",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"failed_types": []},
        )

    invalid_date = df["invoice_ts"].isna()
    invalid_qty = df["quantity"].isna()
    invalid_price = df["unit_price"].isna()
    empty_invoice = df["invoice_no"].isna() | (df["invoice_no"].astype(str).str.strip() == "")
    empty_sku = df["sku"].isna() | (df["sku"].astype(str).str.strip() == "")

    failed_mask = invalid_date | invalid_qty | invalid_price | empty_invoice | empty_sku
    count = int(failed_mask.sum())
    passed = count == 0

    return failed_mask, QualityFinding(
        check_name="type_coercion",
        severity="warning" if count > 0 else "info",
        passed=passed,
        affected_rows=count,
        details={
            "invalid_dates": int(invalid_date.sum()),
            "invalid_quantities": int(invalid_qty.sum()),
            "invalid_prices": int(invalid_price.sum()),
            "empty_invoices": int(empty_invoice.sum()),
            "empty_skus": int(empty_sku.sum()),
        },
    )


def check_cross_sheet_overlap(
    df: pd.DataFrame,
    compare_columns: list[str] | None = None,
) -> tuple[pd.Series, QualityFinding]:
    """Identify cross-sheet duplicate rows per TRD §4.3 Rule 2(a).

    "Drop rows that repeat when keyed by (all fields + occurrence index within their own sheet);
    this removes the second copy without touching legitimate repeats inside a sheet."

    Returns:
        tuple[pd.Series, QualityFinding]: (boolean mask of cross-sheet duplicates, finding)
    """
    if df.empty:
        empty_mask = pd.Series([], dtype=bool)
        return empty_mask, QualityFinding(
            check_name="cross_sheet_overlap",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"cross_sheet_duplicates": 0},
        )

    if "sheet_name" not in df.columns or len(df["sheet_name"].unique()) <= 1:
        # Single sheet or no sheet metadata: no cross-sheet overlap detected
        no_dup_mask = pd.Series(False, index=df.index)
        sheet_list = list(df["sheet_name"].unique()) if "sheet_name" in df.columns else []
        return no_dup_mask, QualityFinding(
            check_name="cross_sheet_overlap",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"cross_sheet_duplicates": 0, "sheets": sheet_list},
        )

    cols = compare_columns or [
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
    cols_present = [c for c in cols if c in df.columns]

    # Calculate occurrence index of each row within its own sheet
    within_sheet_occurrence = df.groupby(["sheet_name", *cols_present]).cumcount()

    # Identify cross-sheet duplicates: same fields + same occurrence index across sheets
    dup_keys = pd.DataFrame(df[cols_present]).assign(_occurrence=within_sheet_occurrence)
    is_cross_sheet_dup = dup_keys.duplicated(keep="first")

    count = int(is_cross_sheet_dup.sum())
    return is_cross_sheet_dup, QualityFinding(
        check_name="cross_sheet_overlap",
        severity="info",
        passed=True,
        affected_rows=count,
        details={"cross_sheet_duplicates": count},
    )


def check_exact_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> tuple[pd.Series, QualityFinding]:
    """Identify exact duplicate rows per TRD §4.3 Rule 2(b).

    Returns:
        tuple[pd.Series, QualityFinding]: (boolean mask of duplicate rows, finding)
    """
    if df.empty:
        empty_mask = pd.Series([], dtype=bool)
        return empty_mask, QualityFinding(
            check_name="exact_duplicates",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"exact_duplicates": 0},
        )

    cols = subset or [
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
    cols_present = [c for c in cols if c in df.columns]
    is_dup = df.duplicated(subset=cols_present, keep="first")
    count = int(is_dup.sum())

    return is_dup, QualityFinding(
        check_name="exact_duplicates",
        severity="info",
        passed=True,
        affected_rows=count,
        details={"exact_duplicates": count},
    )


def classify_returns(df: pd.DataFrame) -> pd.Series:
    """Classify rows as valid returns per TRD §4.3 Rule 3.

    Returns:
        pd.Series: Boolean mask where True indicates a valid return line.
    """
    if df.empty:
        return pd.Series([], dtype=bool)

    invoice_str = df["invoice_no"].astype(str).str.strip().str.upper()
    is_c_invoice = invoice_str.str.startswith("C")
    is_negative_qty = df["quantity"] < 0

    is_c_return = is_c_invoice & is_negative_qty

    has_customer = df["customer_id"].notna() & (df["customer_id"].astype(str).str.strip() != "")
    has_positive_price = df["unit_price"] > 0
    is_non_c_return = (~is_c_invoice) & is_negative_qty & has_customer & has_positive_price

    return is_c_return | is_non_c_return


def identify_invalid_cancellations(df: pd.DataFrame) -> pd.Series:
    """Identify invalid cancellations with positive quantity per TRD §4.3 Rule 3.

    A 'C' invoice with quantity > 0 is invalid and must be rejected.

    Returns:
        pd.Series: Boolean mask where True indicates an invalid cancellation line.
    """
    if df.empty:
        return pd.Series([], dtype=bool)

    invoice_str = df["invoice_no"].astype(str).str.strip().str.upper()
    is_c_invoice = invoice_str.str.startswith("C")
    return is_c_invoice & (df["quantity"] > 0)


def identify_stock_adjustments(df: pd.DataFrame) -> pd.Series:
    """Identify stock adjustments per TRD §4.3 Rule 3.

    Negative quantity without a 'C' invoice is a stock adjustment if customer_id is null
    or unit_price <= 0. These rows are dropped.

    Returns:
        pd.Series: Boolean mask where True indicates a stock adjustment line.
    """
    if df.empty:
        return pd.Series([], dtype=bool)

    invoice_str = df["invoice_no"].astype(str).str.strip().str.upper()
    is_c_invoice = invoice_str.str.startswith("C")
    is_negative_qty = df["quantity"] < 0

    no_customer = df["customer_id"].isna() | (df["customer_id"].astype(str).str.strip() == "")
    no_positive_price = df["unit_price"].isna() | (df["unit_price"] <= 0)

    return (~is_c_invoice) & is_negative_qty & (no_customer | no_positive_price)


def validate_returns_and_cancellations(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, QualityFinding, QualityFinding]:
    """Validate returns, cancellations, and stock adjustments per TRD §4.3 Rule 3.

    Returns:
        tuple[pd.Series, pd.Series, QualityFinding, QualityFinding]:
        (invalid_cancellation_mask, stock_adjustment_mask, invalid_canc_finding, stock_adj_finding)
    """
    invalid_canc_mask = identify_invalid_cancellations(df)
    stock_adj_mask = identify_stock_adjustments(df)

    invalid_canc_count = int(invalid_canc_mask.sum())
    stock_adj_count = int(stock_adj_mask.sum())

    finding_invalid = QualityFinding(
        check_name="invalid_cancellation_qty",
        severity="error" if invalid_canc_count > 0 else "info",
        passed=invalid_canc_count == 0,
        affected_rows=invalid_canc_count,
        details={"invalid_cancellations": invalid_canc_count},
    )

    finding_stock = QualityFinding(
        check_name="stock_adjustments",
        severity="info",
        passed=True,
        affected_rows=stock_adj_count,
        details={"stock_adjustments": stock_adj_count},
    )

    return invalid_canc_mask, stock_adj_mask, finding_invalid, finding_stock


def check_non_product_skus(
    df: pd.DataFrame,
    non_product_skus: set[str] | None = None,
) -> tuple[pd.Series, QualityFinding]:
    """Identify non-product SKUs per TRD §4.3 Rule 4.

    Returns:
        tuple[pd.Series, QualityFinding]: (boolean mask of non-product SKUs, finding)
    """
    if df.empty:
        empty_mask = pd.Series([], dtype=bool)
        return empty_mask, QualityFinding(
            check_name="non_product_skus",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"non_product_skus": 0},
        )

    sku_set = non_product_skus if non_product_skus is not None else load_non_product_skus()
    sku_upper = df["sku"].astype(str).str.strip().str.upper()
    is_non_product = sku_upper.isin(sku_set)
    count = int(is_non_product.sum())

    unique_non_prods = sorted(df.loc[is_non_product, "sku"].unique())
    return is_non_product, QualityFinding(
        check_name="non_product_skus",
        severity="info",
        passed=True,
        affected_rows=count,
        details={
            "non_product_skus": count,
            "unique_non_product_skus": unique_non_prods,
        },
    )


def check_non_positive_price(df: pd.DataFrame) -> tuple[pd.Series, QualityFinding]:
    """Identify lines with unit_price <= 0 per TRD §4.3 Rule 5.

    Returns:
        tuple[pd.Series, QualityFinding]: (boolean mask of non-positive price rows, finding)
    """
    if df.empty:
        empty_mask = pd.Series([], dtype=bool)
        return empty_mask, QualityFinding(
            check_name="non_positive_price",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"non_positive_prices": 0},
        )

    is_non_pos = df["unit_price"].isna() | (df["unit_price"] <= 0)
    count = int(is_non_pos.sum())

    return is_non_pos, QualityFinding(
        check_name="non_positive_price",
        severity="warning" if count > 0 else "info",
        passed=count == 0,
        affected_rows=count,
        details={"non_positive_prices": count},
    )


def check_customer_id_distribution(df: pd.DataFrame) -> QualityFinding:
    """Analyze customer ID presence per TRD §4.3 Rule 6.

    Returns:
        QualityFinding: Accounting for missing and present customer IDs.
    """
    if df.empty:
        return QualityFinding(
            check_name="customer_id_presence",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"total": 0, "missing": 0, "missing_ratio": 0.0},
        )

    missing_mask = df["customer_id"].isna() | (df["customer_id"].astype(str).str.strip() == "")
    missing_count = int(missing_mask.sum())
    total = len(df)
    ratio = missing_count / total if total > 0 else 0.0

    return QualityFinding(
        check_name="customer_id_presence",
        severity="info",
        passed=True,
        affected_rows=missing_count,
        details={"total": total, "missing": missing_count, "missing_ratio": round(ratio, 4)},
    )


def check_hash_uniqueness(hashes: pd.Series) -> QualityFinding:
    """Validate uniqueness of computed source_row_hashes per TRD §4.3 Rule 7.

    Returns:
        QualityFinding: Finding indicating whether row hashes are distinct.
    """
    if hashes.empty:
        return QualityFinding(
            check_name="source_row_hash_uniqueness",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"duplicates": 0},
        )

    dup_count = int(hashes.duplicated().sum())
    passed = dup_count == 0

    return QualityFinding(
        check_name="source_row_hash_uniqueness",
        severity="error" if not passed else "info",
        passed=passed,
        affected_rows=dup_count,
        details={"duplicates": dup_count},
    )


def evaluate_batch_gate(
    total_rows: int,
    rejected_rows: int,
    max_reject_ratio: float = MAX_REJECT_RATIO,
) -> tuple[BatchGateResult, QualityFinding]:
    """Evaluate batch gate per TRD §4.3 Rule 8.

    If more than 20% of input rows are rejected, mark the batch rejected.

    Returns:
        tuple[BatchGateResult, QualityFinding]: (gate result, finding)
    """
    ratio = rejected_rows / total_rows if total_rows > 0 else 0.0
    passed = ratio <= max_reject_ratio

    result = BatchGateResult(
        passed=passed,
        total_rows=total_rows,
        rejected_rows=rejected_rows,
        reject_ratio=ratio,
        max_allowed_ratio=max_reject_ratio,
    )

    finding = QualityFinding(
        check_name="batch_reject_gate",
        severity="error" if not passed else "info",
        passed=passed,
        affected_rows=rejected_rows,
        details={
            "total_rows": total_rows,
            "rejected_rows": rejected_rows,
            "reject_ratio": round(ratio, 4),
            "max_allowed_ratio": max_reject_ratio,
        },
    )

    return result, finding


def enforce_batch_gate(
    total_rows: int,
    rejected_rows: int,
    max_reject_ratio: float = MAX_REJECT_RATIO,
) -> BatchGateResult:
    """Enforce batch gate and raise DataQualityError if threshold is violated."""
    result, _ = evaluate_batch_gate(total_rows, rejected_rows, max_reject_ratio)
    if not result.passed:
        msg = (
            f"Batch gate rejected: {result.rejected_rows}/{result.total_rows} rows rejected "
            f"({result.reject_ratio:.2%}), exceeding threshold {result.max_allowed_ratio:.2%}"
        )
        raise DataQualityError(
            msg,
            details={
                "total_rows": result.total_rows,
                "rejected_rows": result.rejected_rows,
                "reject_ratio": result.reject_ratio,
                "max_allowed_ratio": result.max_allowed_ratio,
            },
        )
    return result


def flag_outliers(
    df: pd.DataFrame,
    outlier_qty: int = OUTLIER_QTY,
) -> tuple[pd.Series, QualityFinding]:
    """Flag lines with |quantity| >= OUTLIER_QTY (50,000) per TRD §4.3 Rule 9.

    Returns:
        tuple[pd.Series, QualityFinding]: (boolean outlier mask, finding)
    """
    if df.empty:
        empty_mask = pd.Series([], dtype=bool)
        return empty_mask, QualityFinding(
            check_name="outlier_quantity",
            severity="info",
            passed=True,
            affected_rows=0,
            details={"outlier_count": 0},
        )

    # Flag both the large order and large return cancellation
    is_outlier = (df["quantity"] >= outlier_qty) | (df["quantity"] <= -outlier_qty)
    count = int(is_outlier.sum())

    outlier_details: list[dict[str, Any]] = []
    if count > 0:
        for idx, row in df[is_outlier].iterrows():
            outlier_details.append(
                {
                    "row_idx": int(idx),
                    "invoice_no": str(row.get("invoice_no", "")),
                    "sku": str(row.get("sku", "")),
                    "quantity": float(row.get("quantity", 0)),
                }
            )

    return is_outlier, QualityFinding(
        check_name="outlier_quantity",
        severity="warning" if count > 0 else "info",
        passed=True,  # Outliers are tracked and flagged, not a fatal rejection of batch
        affected_rows=count,
        details={"outlier_count": count, "outliers": outlier_details},
    )


def validate_schema(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
) -> tuple[bool, list[str]]:
    """Validate DataFrame against a Pandera schema without raising unhandled exceptions.

    Returns:
        tuple[bool, list[str]]: (passed, list of error messages)
    """
    try:
        schema.validate(df, lazy=True)
        return True, []
    except pa.errors.SchemaErrors as err:
        errors = [str(e) for e in err.failure_cases["check"].tolist()]
        return False, errors
    except pa.errors.SchemaError as err:
        return False, [str(err)]


def run_data_quality_checks(
    df: pd.DataFrame,
    non_product_skus: set[str] | None = None,
) -> list[QualityFinding]:
    """Execute complete suite of data quality checks across TRD §4.3 rules.

    Produces findings structured for insertion into ops.data_quality_result.

    Returns:
        list[QualityFinding]: Findings for all checks.
    """
    findings: list[QualityFinding] = []

    # Rule 1: Type coercion
    _, f_coercion = check_type_coercion(df)
    findings.append(f_coercion)

    # Rule 2(a): Cross-sheet overlap
    _, f_overlap = check_cross_sheet_overlap(df)
    findings.append(f_overlap)

    # Rule 2(b): Exact duplicates
    _, f_exact = check_exact_duplicates(df)
    findings.append(f_exact)

    # Rule 3: Returns & cancellations
    _, _, f_invalid_canc, f_stock_adj = validate_returns_and_cancellations(df)
    findings.append(f_invalid_canc)
    findings.append(f_stock_adj)

    # Rule 4: Non-product SKUs
    _, f_non_prod = check_non_product_skus(df, non_product_skus=non_product_skus)
    findings.append(f_non_prod)

    # Rule 5: Non-positive price
    _, f_price = check_non_positive_price(df)
    findings.append(f_price)

    # Rule 6: Customer ID presence
    f_customer = check_customer_id_distribution(df)
    findings.append(f_customer)

    # Rule 9: Outliers
    _, f_outlier = flag_outliers(df)
    findings.append(f_outlier)

    return findings
