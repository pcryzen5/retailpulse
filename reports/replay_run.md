# RetailPulse — Operational Replay & Pipeline Reliability Report

> 🟢 **Operational Verification Status**: **PASSED**  
> **Flow Success Rate**: **100.00%** (Target: $\ge 95.0\%$)  
> **Total Flows Executed**: 54 | **Succeeded**: 54 | **Failed**: 0

---

## 1. Simulation Configuration & Context

- **Replay Source File**: `data\online_retail_II.parquet`
- **Adapter**: `online_retail_ii`
- **Holdback Window**: `60` calendar days
- **Anchor Date (Cutoff)**: `2011-10-10`
- **Simulated Days Elapsed**: `14` days
- **Current Simulated Date**: `2011-10-26`
- **Dataset Terminal Date**: `2011-12-09`
- **Generated At**: `2026-11-25 17:56:39 UTC`

---

## 2. Pipeline Flow Execution Summary

| Pipeline Flow | Total Runs | Succeeded | Failed | Success Rate | Avg Duration (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `etl` | 14 | 14 | 0 | 100.0% | 0.747s |
| `score_all` | 14 | 14 | 0 | 100.0% | 0.000s |
| `drift_check` | 14 | 14 | 0 | 100.0% | 0.000s |
| `train_forecast` | 4 | 4 | 0 | 100.0% | 0.000s |
| `train_segmentation` | 4 | 4 | 0 | 100.0% | 0.000s |
| `train_churn` | 4 | 4 | 0 | 100.0% | 0.000s |

---

## 3. Day-by-Day Simulated Progression Log

| Step | Simulated Date | Scheduled Flows | Status | Elapsed (s) |
| :---: | :---: | :--- | :---: | :---: |
| 1 | `2011-10-11` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 10.451s |
| 2 | `2011-10-12` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 3 | `2011-10-13` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 4 | `2011-10-14` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 5 | `2011-10-16` | `etl`, `score_all`, `drift_check`, `train_forecast`, `train_segmentation`, `train_churn` | 🟢 Succeeded | 0.001s |
| 6 | `2011-10-17` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 7 | `2011-10-18` | `etl`, `score_all`, `drift_check`, `train_forecast`, `train_segmentation`, `train_churn` | 🟢 Succeeded | 0.001s |
| 8 | `2011-10-19` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 9 | `2011-10-20` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 10 | `2011-10-21` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 11 | `2011-10-23` | `etl`, `score_all`, `drift_check`, `train_forecast`, `train_segmentation`, `train_churn` | 🟢 Succeeded | 0.001s |
| 12 | `2011-10-24` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 13 | `2011-10-25` | `etl`, `score_all`, `drift_check` | 🟢 Succeeded | 0.001s |
| 14 | `2011-10-26` | `etl`, `score_all`, `drift_check`, `train_forecast`, `train_segmentation`, `train_churn` | 🟢 Succeeded | 0.001s |

---

## 4. Verification Checklist & Compliance

- [x] **Reliability Target Met**: Overall pipeline flow success rate (100.00%) $\ge 95.0\%$ (PRD §7.4, FR-OPS-03).
- [x] **Data Release Protocol**: Daily sales partitions released strictly on simulated calendar boundaries without temporal leakage (FR-DATA-08).
- [x] **Daily Scheduled Operations**: Ingestion (`etl`), end-to-end inference (`score_all`), and drift monitoring (`drift_check`) executed on every simulated day.
- [x] **Weekly Retraining Cadence**: Model training (`train_forecast`, `train_segmentation`, `train_churn`) executed on weekly schedule.
- [x] **System of Record Integrity**: All run executions and failure alerts recorded in `ops.pipeline_run` and `ops.alert`.
