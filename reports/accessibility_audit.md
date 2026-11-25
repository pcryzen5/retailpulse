# RetailPulse Accessibility Audit & Verification Checklist (DESIGN §8)

This document formalizes the Phase 6 Polish verification audit against the Web Content Accessibility Guidelines (WCAG 2.1 AA standards) and the system design specifications in [docs/DESIGN.md](../docs/DESIGN.md#8-accessibility-checklist).

---

## 1. Executive Summary & Verification Matrix

| Checklist Item (DESIGN §8) | Status | Verification & Evidence |
|:---|:---:|:---|
| **Text contrast ≥ 4.5:1** | **PASSED [x]** | Verified against `#FFFFFF` background: `ink` (17.85:1), `primary` (5.17:1), `muted` (4.76:1), `danger_text` (6.47:1), `warning_text` (5.02:1), and `success_text` (5.02:1). Meets WCAG AA threshold (≥ 4.5:1). |
| **Color-blind safe chart palette** | **PASSED [x]** | Uses official 8-color Okabe-Ito palette in `src/retailpulse/dashboard/theme.py`. Verified for Deuteranopia, Protanopia, and Tritanopia discernibility. |
| **Status/risk never conveyed by color alone** | **PASSED [x]** | Standardized via `src/retailpulse/dashboard/components/badges.py`. Every badge couples an explicit text label with a distinct Unicode symbol: OK `✔`, Reorder `⚠`, Overstock `◔`, High Risk `●`, Synthetic `🧪`, Queued `⏳`, Succeeded `✔`, Failed `❌`. |
| **Every chart has title, axis labels & text summary** | **PASSED [x]** | Enforced by `render_chart(fig, caption=...)` helper across all views (`overview`, `forecast`, `segments`, `churn`, `monitoring`). Always renders an accessible caption beneath the Plotly container. |
| **Tables have clear column headers & no hover-only data** | **PASSED [x]** | All tabular displays use native `st.dataframe` with fully defined column names and right-aligned metrics. Critical insights and SHAP reasons are surfaced as discrete table columns and expandable cards, not hover tooltips. |
| **All actions reachable by keyboard** | **PASSED [x]** | Streamlit standard native form inputs (`st.selectbox`, `st.button`, `st.text_input`, `st.checkbox`, `st.file_uploader`) adhere to standard WAI-ARIA tab stops and keyboard focus indicators. |
| **No auto-playing motion; switchable auto-refresh** | **PASSED [x]** | Auto-refresh is strictly isolated to Overview and Pipelines pages, defaulting to `False`, and provided with a clear toggle switch (`st.checkbox("Auto-refresh (60s)")`). Zero animated carousels or autoplay media. |
| **Plain-language microcopy & tooltips for metrics** | **PASSED [x]** | Centralized metric definition repository in `src/retailpulse/dashboard/components/kpi_card.py` (`METRIC_TOOLTIPS`). WAPE, MASE, Coverage, Safety Stock, Reorder Point, and PSI provide explanatory help text. |

---

## 2. Token Contrast Measurements

Calculated standard relative luminance and contrast ratio against default Surface `#FFFFFF`:

```text
- Primary (#2563EB)     :  5.17:1  (Target: ≥ 4.5:1 -> PASS)
- Ink (#0F172A)         : 17.85:1  (Target: ≥ 4.5:1 -> PASS)
- Muted (#64748B)       :  4.76:1  (Target: ≥ 4.5:1 -> PASS)
- Success Text (#15803D):  5.02:1  (Target: ≥ 4.5:1 -> PASS)
- Warning Text (#B45309):  5.02:1  (Target: ≥ 4.5:1 -> PASS)
- Danger Text (#B91C1C) :  6.47:1  (Target: ≥ 4.5:1 -> PASS)
```

---

## 3. Visual Assets & Screenshots for README

The following high-resolution UI preview assets have been generated in `docs/assets/screenshots/` and referenced in `README.md`:

1. `overview.png`: Executive KPI strip, 30-day demand trends, top stockout risks, and segment breakdown.
2. `demand_forecast.png`: SKU search selector, 7/14/28-day horizon toggle, 180-day history with 80% prediction interval.
3. `customer_segments.png`: RFM segment share cards, grouped bar distributions, and churn cross-navigation.
4. `churn_risk.png`: Churn probability histogram, plain-language SHAP driver explanations, and CSV export.
5. `inventory_policy.png`: Multi-echelon inventory parameters, what-if parameter delta simulator, and policy comparison.
6. `model_monitoring.png`: 4-tab model registry, candidate comparison, PSI feature drift, and sandboxed HTML viewer.
7. `data_pipelines.png`: Sales CSV upload panel, batch data quality validation, and execution trigger controls.
8. `admin_users.png`: System account administration, role assignment controls, and last-active-admin protection.
