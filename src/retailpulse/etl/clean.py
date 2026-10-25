"""ETL cleaning pipeline implementing TRD §4.3 rules 1-7 and 9.

Applies sequential cleaning rules:
  1. String trimming, type coercion, and quarantine separation (Rule 1)
  2. Cross-sheet overlap deduplication (Rule 2a) and exact row deduplication (Rule 2b)
  3. Return/cancellation polarity validation and stock adjustment dropping (Rule 3)
  4. Non-product SKU classification from explicit configuration (Rule 4)
  5. Dropping non-positive unit prices (Rule 5)
  6. HMAC-SHA256 customer pseudonymization and removal of raw customer IDs (Rule 6)
  7. Deterministic source_row_hash generation for idempotent loading (Rule 7)
  8. Batch rejection gate evaluation (Rule 8)
  9. Outlier quantity flagging (Rule 9)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from pydantic import SecretStr

from retailpulse.config import get_settings
from retailpulse.errors import DataQualityError
from retailpulse.etl.quality import (
    OUTLIER_QTY,
    QualityFinding,
    check_cross_sheet_overlap,
    check_customer_id_distribution,
    check_exact_duplicates,
    check_hash_uniqueness,
    check_non_positive_price,
    check_non_product_skus,
    check_type_coercion,
    classify_returns,
    cleaned_sales_line_schema,
    evaluate_batch_gate,
    flag_outliers,
    identify_invalid_cancellations,
    identify_stock_adjustments,
    load_non_product_skus,
    validate_schema,
)
from retailpulse.ingestion.schemas import (
    canonicalize_customer_id,
    canonicalize_invoice_no,
    canonicalize_stock_code,
)
from retailpulse.logging import get_logger
from retailpulse.security.pseudonymize import pseudonymize_series

logger = get_logger(__name__)

# ASCII unit separator for canonical hashing per TRD §4.3 Rule 7
UNIT_SEPARATOR: str = "\x1f"

# Final clean DataFrame column order matching core.fact_sales_line
CLEANED_COLUMNS: list[str] = [
    "invoice_no",
    "invoice_ts",
    "sku",
    "description",
    "quantity",
    "unit_price",
    "customer_key",
    "country",
    "store_code",
    "is_return",
    "is_product",
    "is_outlier",
    "source_row_hash",
]


@dataclass(frozen=True)
class CleanedDataset:
    """Result of running the ETL cleaning pipeline."""

    df: pd.DataFrame
    quarantined_df: pd.DataFrame
    findings: list[QualityFinding]
    row_counts: dict[str, int] = field(default_factory=dict)


def compute_source_row_hash(
    invoice_no: str,
    invoice_ts: pd.Timestamp | str,
    sku: str,
    *,
    description: str | None = None,
    quantity: int | float = 0,
    unit_price: float = 0.0,
    customer_key: str | None = None,
    country: str = "United Kingdom",
    store_code: str = "ONLINE",
) -> str:
    """Compute sha256 hash of canonical fields joined by a unit separator per TRD §4.3 Rule 7.

    Returns:
        64-character lowercase hex SHA-256 hash string.
    """
    ts_str = (
        invoice_ts.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(invoice_ts, pd.Timestamp)
        else str(invoice_ts).strip()
    )
    desc_str = str(description).strip() if description is not None and pd.notna(description) else ""
    qty_str = str(int(quantity))
    price_str = f"{float(unit_price):.4f}"
    cust_str = (
        str(customer_key).strip() if customer_key is not None and pd.notna(customer_key) else ""
    )

    parts = [
        str(invoice_no).strip(),
        ts_str,
        str(sku).strip().upper(),
        desc_str,
        qty_str,
        price_str,
        cust_str,
        str(country).strip(),
        str(store_code).strip(),
    ]
    payload = UNIT_SEPARATOR.join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_source_row_hashes(df: pd.DataFrame) -> pd.Series:
    """Compute deterministic source_row_hashes across an entire DataFrame.

    Returns:
        Series of 64-character hex strings indexed like df.
    """
    if df.empty:
        return pd.Series([], dtype="object", index=df.index)

    hashes = [
        compute_source_row_hash(
            invoice_no=row["invoice_no"],
            invoice_ts=row["invoice_ts"],
            sku=row["sku"],
            description=row.get("description"),
            quantity=row["quantity"],
            unit_price=row["unit_price"],
            customer_key=row.get("customer_key"),
            country=row.get("country", "United Kingdom"),
            store_code=row.get("store_code", "ONLINE"),
        )
        for _, row in df.iterrows()
    ]
    return pd.Series(hashes, index=df.index, dtype="object")


def _apply_coercion_step(
    df: pd.DataFrame,
    quarantined_records: list[pd.DataFrame],
    findings: list[QualityFinding],
) -> pd.DataFrame:
    """Normalize string columns and separate type coercion failures per Rule 1."""
    df["invoice_no"] = df["invoice_no"].apply(canonicalize_invoice_no)
    df["sku"] = df["sku"].apply(canonicalize_stock_code)
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].apply(canonicalize_customer_id)

    failed_coercion_mask, finding_coercion = check_type_coercion(df)
    findings.append(finding_coercion)

    if failed_coercion_mask.any():
        bad_rows = df[failed_coercion_mask].copy().assign(quarantine_reason="type_coercion")
        quarantined_records.append(bad_rows)
        return df[~failed_coercion_mask].copy()
    return df


def _apply_deduplication_step(
    df: pd.DataFrame,
    dedupe_exact_rows: bool,
    findings: list[QualityFinding],
) -> pd.DataFrame:
    """Apply cross-sheet overlap deduplication (2a) and exact row deduplication (2b)."""
    is_cross_sheet_dup, finding_overlap = check_cross_sheet_overlap(df)
    findings.append(finding_overlap)
    if is_cross_sheet_dup.any():
        df = df[~is_cross_sheet_dup].copy()

    if dedupe_exact_rows:
        is_exact_dup, finding_exact = check_exact_duplicates(df)
        findings.append(finding_exact)
        if is_exact_dup.any():
            df = df[~is_exact_dup].copy()

    return df


def _apply_cancellation_and_adjustment_step(
    df: pd.DataFrame,
    quarantined_records: list[pd.DataFrame],
    findings: list[QualityFinding],
) -> pd.DataFrame:
    """Enforce return/cancellation polarity and drop stock adjustments per Rule 3."""
    invalid_canc_mask = identify_invalid_cancellations(df)
    finding_invalid_canc = QualityFinding(
        check_name="invalid_cancellation_qty",
        severity="error" if invalid_canc_mask.any() else "info",
        passed=not invalid_canc_mask.any(),
        affected_rows=int(invalid_canc_mask.sum()),
        details={"invalid_cancellations": int(invalid_canc_mask.sum())},
    )
    findings.append(finding_invalid_canc)

    if invalid_canc_mask.any():
        bad_canc = df[invalid_canc_mask].copy().assign(quarantine_reason="invalid_cancellation_qty")
        quarantined_records.append(bad_canc)
        df = df[~invalid_canc_mask].copy()

    stock_adj_mask = identify_stock_adjustments(df)
    finding_stock_adj = QualityFinding(
        check_name="stock_adjustments",
        severity="info",
        passed=True,
        affected_rows=int(stock_adj_mask.sum()),
        details={"stock_adjustments": int(stock_adj_mask.sum())},
    )
    findings.append(finding_stock_adj)
    if stock_adj_mask.any():
        df = df[~stock_adj_mask].copy()

    df["is_return"] = classify_returns(df)
    return df


def _apply_pseudonymization_and_outliers(
    df: pd.DataFrame,
    salt: str | SecretStr | None,
    findings: list[QualityFinding],
) -> pd.DataFrame:
    """Pseudonymize customer ID, drop raw ID, flag outliers, and cast types."""
    finding_cust = check_customer_id_distribution(df)
    findings.append(finding_cust)
    df["customer_key"] = pseudonymize_series(df["customer_id"], salt=salt)
    df = df.drop(columns=["customer_id"])

    for prov_col in ["sheet_name", "sheet_row_idx"]:
        if prov_col in df.columns:
            df = df.drop(columns=[prov_col])

    is_outlier, finding_outlier = flag_outliers(df, outlier_qty=OUTLIER_QTY)
    findings.append(finding_outlier)
    df["is_outlier"] = is_outlier

    df["quantity"] = df["quantity"].astype("int64")
    df["unit_price"] = df["unit_price"].astype("float64")
    df["is_return"] = df["is_return"].astype(bool)
    df["is_product"] = df["is_product"].astype(bool)
    df["is_outlier"] = df["is_outlier"].astype(bool)

    df["source_row_hash"] = compute_source_row_hashes(df)
    finding_hash = check_hash_uniqueness(df["source_row_hash"])
    findings.append(finding_hash)
    return df


def _evaluate_batch_and_quarantine(
    initial_count: int,
    quarantined_records: list[pd.DataFrame],
    enforce_threshold: bool,
    quarantine_dir: Path | str | None,
    findings: list[QualityFinding],
) -> pd.DataFrame:
    """Evaluate batch rejection gate and optionally persist quarantine file."""
    quarantined_df = (
        pd.concat(quarantined_records, ignore_index=True) if quarantined_records else pd.DataFrame()
    )
    total_rejected = len(quarantined_df)

    gate_result, finding_gate = evaluate_batch_gate(
        total_rows=initial_count,
        rejected_rows=total_rejected,
    )
    findings.append(finding_gate)

    if enforce_threshold and not gate_result.passed:
        rej_n = gate_result.rejected_rows
        tot_n = gate_result.total_rows
        ratio_pct = gate_result.reject_ratio
        lim_pct = gate_result.max_allowed_ratio
        msg = (
            f"Batch gate rejected: {rej_n}/{tot_n} rows rejected "
            f"({ratio_pct:.2%}), exceeding limit {lim_pct:.2%}"
        )
        logger.error("batch gate rejected dataset", details=finding_gate.details)
        raise DataQualityError(msg, details=finding_gate.details)

    if not quarantined_df.empty:
        target_dir = (
            Path(quarantine_dir)
            if quarantine_dir
            else get_settings().data_dir / "interim" / "quarantine"
        )
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            quarantine_file = target_dir / "quarantine_batch.csv"
            quarantined_df.to_csv(quarantine_file, index=False)
            logger.info(
                "wrote quarantined records",
                path=str(quarantine_file),
                rows=len(quarantined_df),
            )
        except Exception as exc:
            logger.warning("failed to write quarantine file", error=str(exc))

    return quarantined_df


def clean_sales_lines(
    raw_df: pd.DataFrame,
    *,
    salt: str | SecretStr | None = None,
    dedupe_exact_rows: bool = True,
    non_product_skus: set[str] | None = None,
    enforce_batch_gate_threshold: bool = True,
    quarantine_dir: Path | str | None = None,
) -> CleanedDataset:
    """Execute complete ETL cleaning pipeline per TRD §4.3 rules 1-7 and 9.

    Args:
        raw_df: Canonical DataFrame from ingestion adapter.
        salt: Optional salt for customer pseudonymization. Defaults to Settings.
        dedupe_exact_rows: Whether to drop exact duplicate rows (Rule 2b).
        non_product_skus: Optional explicit non-product SKU set. Defaults to config file.
        enforce_batch_gate_threshold: Whether to raise DataQualityError if reject ratio > 20%.
        quarantine_dir: Optional path to write quarantined failure rows.

    Returns:
        CleanedDataset containing cleaned DataFrame, quarantined rows, findings, and counts.
    """
    findings: list[QualityFinding] = []
    row_counts: dict[str, int] = {}
    quarantined_records: list[pd.DataFrame] = []

    initial_count = len(raw_df)
    row_counts["raw_read"] = initial_count
    logger.info("starting sales line cleaning", total_raw_rows=initial_count)

    if raw_df.empty:
        empty_clean = pd.DataFrame(columns=CLEANED_COLUMNS)
        return CleanedDataset(
            df=empty_clean,
            quarantined_df=raw_df.copy(),
            findings=[],
            row_counts=row_counts,
        )

    # Steps 1 to 3: Coercion, deduplication, cancellation & adjustments
    df = _apply_coercion_step(raw_df.copy(), quarantined_records, findings)
    row_counts["after_coercion"] = len(df)

    df = _apply_deduplication_step(df, dedupe_exact_rows, findings)
    row_counts["after_deduplication"] = len(df)

    df = _apply_cancellation_and_adjustment_step(df, quarantined_records, findings)
    row_counts["after_returns_and_adjustments"] = len(df)

    # Step 4: Non-product SKUs
    sku_set = non_product_skus if non_product_skus is not None else load_non_product_skus()
    is_non_product, finding_non_prod = check_non_product_skus(df, non_product_skus=sku_set)
    findings.append(finding_non_prod)
    df["is_product"] = ~is_non_product

    # Step 5: Non-positive unit prices
    is_non_pos_price, finding_price = check_non_positive_price(df)
    findings.append(finding_price)
    if is_non_pos_price.any():
        df = df[~is_non_pos_price].copy()
    row_counts["after_non_positive_price"] = len(df)

    # Steps 6, 9, 7: Pseudonymization, outliers, row hashes
    df = _apply_pseudonymization_and_outliers(df, salt, findings)

    # Step 8: Batch Gate & Quarantine
    quarantined_df = _evaluate_batch_and_quarantine(
        initial_count,
        quarantined_records,
        enforce_batch_gate_threshold,
        quarantine_dir,
        findings,
    )
    row_counts["quarantined_total"] = len(quarantined_df)

    # Reorder columns to match canonical schema order
    df = df[CLEANED_COLUMNS]
    row_counts["final_cleaned"] = len(df)

    is_valid, schema_errors = validate_schema(df, cleaned_sales_line_schema)
    if not is_valid:
        raise DataQualityError(
            f"Cleaned DataFrame failed cleaned_sales_line_schema validation: {schema_errors}",
            details={"schema_errors": schema_errors},
        )

    logger.info("cleaning pipeline finished successfully", row_counts=row_counts)
    return CleanedDataset(
        df=df,
        quarantined_df=quarantined_df,
        findings=findings,
        row_counts=row_counts,
    )
