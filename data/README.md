# RetailPulse — Dataset Guide & Data Hygiene

This document provides instructions for obtaining, preparing, and loading the primary dataset for RetailPulse: **Online Retail II** from the UCI Machine Learning Repository.

---

## 1. Primary Dataset Overview

- **Dataset:** Online Retail II
- **Repository:** UCI Machine Learning Repository (Dataset ID: 502)
- **Direct UCI URL:** [https://archive.ics.uci.edu/dataset/502/online+retail+ii](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
- **DOI:** [10.24432/C5CG6D](https://doi.org/10.24432/C5CG6D)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Citation:**
  > Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

The dataset contains all transactions occurring between **2009-12-01** and **2011-12-09** for a UK-based non-store online retail business selling unique all-occasion gift-ware.

---

## 2. Obtaining the Data

1. Download the official archive from UCI:
   ```bash
   curl -LO "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
   ```
2. Unzip the file:
   ```bash
   unzip "online+retail+ii.zip"
   ```
   This extracts `online_retail_II.xlsx` (≈ 43.5 MB).
3. Place `online_retail_II.xlsx` in the workspace root or under `data/raw/`.

---

## 3. Workbook Structure & Columns

The workbook contains two distinct sheets, covering two full operational years:
- **`Year 2009-2010`**: 525,461 rows (transactions from 2009-12-01 to 2010-12-09)
- **`Year 2010-2011`**: 541,910 rows (transactions from 2010-12-01 to 2011-12-09)
- **Total Raw Lines:** **1,067,371** rows

### Column Aliases & Mapping
Different mirrors or exports of this dataset use slight column name variations. The RetailPulse adapter automatically detects and normalizes:
| Canonical Field | Raw Aliases Detected | Description |
|---|---|---|
| `invoice_no` | `Invoice`, `InvoiceNo` | 6-digit transaction identifier; cancellation starts with `C` |
| `sku` | `StockCode`, `Stock Code` | Product identifier (text, trimmed, uppercased) |
| `description` | `Description` | Product name / catalog description |
| `quantity` | `Quantity` | Transaction units (+ for sale, - for return) |
| `invoice_ts` | `InvoiceDate`, `Invoice Date` | Local transaction timestamp (timezone-naive) |
| `unit_price` | `Price`, `UnitPrice` | Unit selling price in GBP |
| `customer_id` | `Customer ID`, `CustomerID` | Numeric buyer identifier (contains ~22.8% nulls) |
| `country` | `Country` | Country name of the customer |

---

## 4. Data Quirks & Cleaning Funnel (TRD §4.3)

During ETL ingestion (`retailpulse etl run`), the pipeline applies the following cleaning rules in strict order:

1. **Cross-Sheet Overlap (Rule 2a):**
   - The two sheets overlap from **2010-12-01 to 2010-12-09** (**22,523 rows** duplicated between the two sheets).
   - Removed by keyed deduplication across sheet occurrences.
   - Rows remaining: **1,044,848**.
2. **Exact Duplicate Lines (Rule 2b):**
   - Remaining exact duplicate rows (**11,812 rows**) are removed.
   - Rows remaining: **1,033,036**.
3. **Returns vs. Stock Adjustments (Rule 3):**
   - `C` invoice prefix + negative quantity = valid return.
   - Negative quantity without `C` invoice and without customer ID = warehouse stock adjustment (**3,457 rows** dropped).
   - `C` invoice prefix with positive quantity = invalid cancellation (**1 row** quarantined).
   - Rows remaining: **1,029,578**.
4. **Non-Positive Prices (Rule 5):**
   - Lines with `price <= 0` (**2,120 rows**) are dropped and logged.
   - Rows remaining: **1,027,458**.
5. **Non-Product SKUs (Rule 4):**
   - SKUs listed in `config/non_product_skus.yaml` (`POST`, `DOT`, `M`, `BANK CHARGES`, etc.) are retained in `fact_sales_line` with `is_product = false` (excluded from demand forecasts).
   - Note: `DCGS*` codes are verified real products and are kept as products.
6. **Outlier Flags (Rule 9):**
   - Lines with `quantity >= 50,000` are flagged with `is_outlier = true` (two orders: 80,995 and 74,215 units).
   - Preserved in `fact_sales_line`, but excluded from `fact_daily_sales` and customer analytics.
7. **Customer Pseudonymization (Rule 6):**
   - Raw customer IDs are converted to `customer_key = HMAC_SHA256(salt, canonical_id)`. Raw IDs never enter the database.
8. **Incomplete Final Day (TRD §4.4):**
   - The final transaction date, **2011-12-09**, ends prematurely at 12:50 UTC (flagged as `is_complete = false` in `dim_date`).
   - The platform anchor date (`data_as_of`) is set to the last complete trading day: **2011-12-08**.

---

## 5. Converting to Parquet (Recommended)

Reading `online_retail_II.xlsx` with OpenPyXL takes ~90–120 seconds. Converting once to Parquet reduces load time to **~1.5 seconds**.

### Conversion Command:
```bash
uv run python scripts/convert_to_parquet.py --input online_retail_II.xlsx --output data/online_retail_II.parquet
```

---

## 6. Running Ingestion & Aggregation

Run the full ETL pipeline with the CLI:
```bash
# From Parquet (fastest):
uv run retailpulse etl run --source online_retail_ii --path data/online_retail_II.parquet

# Or directly from the Excel workbook:
uv run retailpulse etl run --source online_retail_ii --path online_retail_II.xlsx
```

---

## 7. Directory Layout

- `data/raw/`: Original source data archives and raw spreadsheets.
- `data/interim/`: Intermediate outputs and quarantined rows (`data/interim/quarantine/`).
- `data/processed/`: Parquet files and derived artifacts.
