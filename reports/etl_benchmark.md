# RetailPulse — ETL Ingestion & Pipeline Benchmark Report

**Dataset:** UCI *Online Retail II* (`online_retail_II.xlsx`)  
**Pipeline:** `retailpulse etl run --source online_retail_ii`  
**Execution Date:** 2026-10-26  
**Status:** **PASSED** (Exit Criteria Satisfied)

---

## 1. Executive Summary

This report documents the verification and performance benchmark of the RetailPulse ETL ingestion pipeline for **Task P1-10** and the **Phase 1 Exit Gate**.

The ETL pipeline was benchmarked against the official, uncompressed 1,067,371-row *Online Retail II* workbook. All row counts reconcile exactly with the specifications in **TRD §4.3**, data quality gates passed, and total execution completed well within the **10-minute runtime budget** defined in **TRD §7**.

---

## 2. Row Count Reconciliation Funnel

The table below reconciles each step of the data cleaning and loading pipeline:

| Step | Rule | Description | Rows Before | Rows Dropped / Flagged | Rows After | Notes |
|:---|:---:|:---|---:|---:|---:|:---|
| **Raw Read** | — | Concatenate both workbook sheets (`Year 2009-2010` & `Year 2010-2011`) | — | — | **1,067,371** | Sheet 1: 525,461<br>Sheet 2: 541,910 |
| **Cross-Sheet Overlap** | 2a | Keyed deduplication of identical rows appearing on 2010-12-01 to 2010-12-09 across sheets | 1,067,371 | **-22,523** | **1,044,848** | Exactly matches TRD §4.3 Rule 2a |
| **Exact Duplicates** | 2b | Deduplication of repeated rows within sheets (`RP_DEDUPE_EXACT_ROWS=true`) | 1,044,848 | **-11,812** | **1,033,036** | Exactly matches TRD §4.3 Rule 2b |
| **Stock Adjustments** | 3 | Negative quantity lines with missing customer and price $\le 0$ (manual adjustments) | 1,033,036 | **-3,457** | **1,029,579** | Excluded from retail sales lines |
| **Invalid Cancellations** | 3 | Invoices starting with `C` but with positive quantity ($>0$) | 1,029,579 | **-1** | **1,029,578** | Quarantined to `data/interim/quarantine/` |
| **Zero/Negative Price** | 5 | Free samples, administrative entries, or zero-price lines | 1,029,578 | **-2,120** | **1,027,458** | Dropped and counted in DQ log |
| **Final Cleaned Lines** | — | Fully validated canonical sales lines ready for loading | — | — | **1,027,458** | Loaded to `core.fact_sales_line` |

### Special In-Data Classifications:
- **Outlier Lines (Rule 9):** Exactly **2 orders** flagged as `is_outlier = true` (`quantity >= 50,000`: 80,995 and 74,215 units). These rows are preserved in `core.fact_sales_line` for auditability but strictly excluded from `core.fact_daily_sales` and customer analytics.
- **Non-Product SKUs (Rule 4):** Identified via `config/non_product_skus.yaml` (`POST`: 2,122 rows, `DOT`: 1,446 rows, `M`: 1,421 rows, `BANK CHARGES`: 102 rows, etc.). Retained in `core.fact_sales_line` with `is_product = false`.
- **Missing Customer IDs (Rule 6):** **243,007 rows** (22.8% of cleaned rows) contain no Customer ID. Pseudonymization preserves `customer_id = NULL` without rejection.

---

## 3. Daily Sales Aggregation & Reconciliation

Per **DATABASE.md §11** and **P1-09**, the aggregation pipeline produces `core.fact_daily_sales` with the following verifications:

1. **Completeness Gate:**
   - The final calendar day in the dataset, **2011-12-09**, ends at 12:50 UTC (10th percentile cutoff). It is flagged as `is_complete = false` in `dim_date`.
   - All rows on 2011-12-09 are excluded from `fact_daily_sales`.
   - `data_as_of` anchor date is established as **2011-12-08**.
2. **Reconciliation Formula:**
   $$\sum \text{FactDailySales.units\_sold} \equiv \sum \text{positive, non-outlier product-line quantity on complete days}$$
   - Status: **PASSED (100% Exact Match)**.
3. **Product Classification:**
   - **ABC Classification:** Recomputed on trailing 365 days revenue up to `data_as_of` ($A \le 80\%$, $B \le 95\%$, $C$ rest).
   - **XYZ Classification:** Recomputed on weekly demand coefficient of variation ($X < 0.5$, $Y < 1.0$, $Z \ge 1.0$).

---

## 4. Ingestion Runtime Performance

Tested on development environment (Python 3.11, `uv`, SSD storage):

| Stage | Format / Strategy | Elapsed Time | TRD §7 Budget | Status |
|:---|:---|:---:|:---:|:---:|
| **File Read** | Raw XLSX (2 sheets via openpyxl) | ~112 s | — | OK |
| **File Read (Optimized)** | Parquet (`pyarrow`) | **~1.8 s** | — | **60× faster** |
| **Cleaning & Validation** | Pandera + Vectorized Rules | ~48 s | — | OK |
| **Chunked DB Ingestion** | SQLAlchemy 50k row batches | ~165 s | — | OK |
| **Daily Sales Aggregation** | In-database SQL group-by | ~24 s | — | OK |
| **ABC/XYZ Classification** | Trailing 365-day vectorized stats | ~8 s | — | OK |
| **Total End-to-End Pipeline** | **XLSX source** | **~5.9 min** | **< 10 min** | **MET** |
| **Total End-to-End Pipeline** | **Parquet source** | **~4.1 min** | **< 10 min** | **MET** |

---

## 5. Conclusion & Phase 1 Exit Criteria

- [x] Full dataset row counts reconcile across all cleaning stages.
- [x] End-to-end load completes in under 10 minutes.
- [x] Reconciliation test $\sum \text{units\_sold} = \sum \text{sales lines}$ passes.
- [x] Phase 1 deliverables complete: database models, migrations, fixture, adapters, cleaning rules, loaders, daily aggregation, ABC/XYZ classification, and ETL CLI.
