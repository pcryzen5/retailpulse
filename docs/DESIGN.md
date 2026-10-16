# RetailPulse — UI/UX Design

**Version:** 1.0 | **UI framework:** Streamlit + Plotly | **Related:** [APP_FLOW](APP_FLOW.md) · [API](API.md) · [PRD](PRD.md) · [SECURITY](SECURITY.md)

This file defines what the dashboard looks like and how components behave. Navigation and user journeys are in APP_FLOW.md. The dashboard is a pure API client (ARCHITECTURE ADR-006).

---

## 1. Design principles

1. **Decision first.** Every page answers one question ("What should I reorder?", "Who is likely to leave?"). The answer is above the fold.
2. **Honest uncertainty.** Forecasts always show their interval. Synthetic inventory parameters always carry a visible "Synthetic" badge.
3. **Explain, don't just score.** Churn and stockout results show plain-language reasons.
4. **Calm and consistent.** One palette, one layout grid, one set of components. No decoration without information.
5. **Fast by default.** Server-side pagination, cached reads, spinners only where work is real.
6. **Accessible.** No information by color alone; WCAG 2.1 AA contrast targets; text summaries for charts.

## 2. Design tokens

### Color (light theme)
| Token | Hex | Use |
|-------|-----|-----|
| `primary` | `#2563EB` | Primary actions, links, selected state |
| `ink` | `#0F172A` | Body text |
| `muted` | `#64748B` | Secondary text, captions (use ≥ 14 px) |
| `border` | `#E2E8F0` | Dividers, card borders |
| `surface` | `#FFFFFF` | Cards, page |
| `surface_alt` | `#F1F5F9` | Sidebar, table headers |
| `success` / `success_text` | `#16A34A` / `#15803D` | Fills and borders / text |
| `warning` / `warning_text` | `#F59E0B` / `#B45309` | Fills and borders / text |
| `danger` / `danger_text` | `#DC2626` / `#B91C1C` | Fills and borders / text |
| `info` | `#2563EB` | Informational badges |

Never use the plain `success` or `warning` colors as text on white (insufficient contrast); use the `*_text` variants.

### Chart palette (color-blind safe, Okabe-Ito order)
`#0072B2`, `#E69F00`, `#009E73`, `#D55E00`, `#CC79A7`, `#56B4E9`, `#F0E442`, `#000000`. At most 6 series per chart; beyond that, aggregate or facet.

| Semantic mapping | Colors |
|------------------|--------|
| Segments | Champions `#0072B2`, Loyal `#009E73`, New / Promising `#56B4E9`, Big Spenders `#E69F00`, At Risk `#D55E00`, Hibernating `#6B7280` |
| Risk bands | Low `#16A34A`, Medium `#F59E0B`, High `#DC2626` (always with a text label) |
| Forecast | History solid `#0072B2`; forecast dashed `#D55E00`; interval band same hue at 20% opacity; baseline `#6B7280` dotted |

### Typography and spacing
- Font: `sans serif` (Streamlit default; Inter if available). H1 28 px, H2 22, H3 18, body 14–16, caption 12–13.
- 8-pt spacing grid. Cards: 16 px padding, 1 px `border`, 8 px radius.
- Locale `en-GB`: `£12,345.50`, KPI cards compact (`£612k`, `£1.2M`), units with thousands separators, percentages with 1 decimal, dates `09 Dec 2011`. Currency symbol comes from `RP_CURRENCY`.

### Streamlit theme (`.streamlit/config.toml`)
```toml
[theme]
base = "light"
primaryColor = "#2563EB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F1F5F9"
textColor = "#0F172A"
font = "sans serif"

[browser]
gatherUsageStats = false

[server]
headless = true
```
Dark mode is a P1 nice-to-have: keep `st.plotly_chart` on its default Streamlit theme so fonts and backgrounds follow the active theme, and set explicit trace colors from the palette above.

## 3. Global layout

```
+----------------------------------------------------------------------------------+
| RetailPulse   Data through 08 Dec 2011 | Champion: LightGBM | Pipeline: OK 01:22 UTC|  status strip
+----------------+-----------------------------------------------------------------+
| Overview       |  Page title                                  [Refresh] [Export] |
| Demand Forecast|  -------------------------------------------------------------- |
| Segments       |  Filter bar                                                     |
| Churn Risk     |  -------------------------------------------------------------- |
| Inventory      |  Content                                                        |
| Monitoring     |                                                                 |
| Data & Pipelines|                                                                |
| Admin (admin)  |                                                                 |
| user · logout  |                                                                 |
+----------------+-----------------------------------------------------------------+
```

- **Status strip** (every page): `data_as_of`, champion forecast model, last pipeline status and time. Turns into a warning banner if the last successful `score_all` is older than 36 hours.
- **Navigation:** `st.navigation` with a role-based page list (viewers never see Admin; write actions are hidden, not disabled).
- **Layout:** `layout="wide"`; content in a 12-column mental grid using `st.columns`.
- **Primary desktop target:** ≥ 1280 px. Tablets work; on narrow screens columns stack and tables scroll horizontally.

## 4. Page specifications

Data sources refer to endpoints in API.md.

### 4.1 Login
Email, password, Sign in. Generic error "Email or password is incorrect." Locked accounts show "Too many attempts. Try again later." No password hints, no "remember me" in v1.

### 4.2 Overview (viewer+)
```
[Revenue 30d £612k ▲2.4%] [Active customers 1,840] [Forecast WAPE 0.42 vs 0.51] [SKUs to reorder 37] [High-risk customers 214] [Revenue at risk £48k]
+-------------------------------------------+  +--------------------------------+
| Demand (units, top-N SKUs): history +      |  | Alerts (open: 2)               |
| 28-day forecast with 80% band              |  | ⚠ Realized WAPE 0.63 > ... [Ack]|
+-------------------------------------------+  +--------------------------------+
+--------------------------------+  +--------------------------------------------+
| Top 10 stockout risks (table)  |  | Customers by segment: share of customers   |
|                                |  | vs share of revenue (grouped bars)         |
+--------------------------------+  +--------------------------------------------+
```
Sources: `/kpis/overview`, `/forecasts` (aggregate via `/sales/daily` for history), `/inventory/recommendations?status=reorder_now&page_size=10`, `/segments`, `/alerts?status=open`.
KPI deltas: arrow + sign + text ("▲ 2.4% vs previous 30 days"), color is secondary.

### 4.3 Demand Forecast (viewer+)
```
SKU [search/select ▼]  Store [ONLINE ▼]  Horizon [7 | 14 | 28]        Model: LightGBM · trained 04 Oct
+--------------------------------------------------------------------------+
| History (last 180 days) + forecast + 80% interval band + baseline toggle |
+--------------------------------------------------------------------------+
| Next 28 days: 412 units (260–590) | ABC class A | WAPE for class A: 0.31   |
+-------------------------------------+------------------------------------+
| Accuracy by ABC: champion vs        | Accuracy by horizon bucket         |
| seasonal naive (grouped bars)       | (1–7, 8–14, 15–28)                 |
+-------------------------------------+------------------------------------+
| Forecast table: date · forecast · lower · upper        [Download CSV]      |
```
Sources: `/products?q=`, `/sales/daily`, `/forecasts`, `/forecasts/accuracy`. A non-eligible SKU shows the reason returned by the API, not an error.

### 4.4 Customer Segments (viewer+)
Segment cards (label, customers, share of revenue, avg R/F/M, one-sentence plain-language description). Below: grouped bar (customer share vs revenue share) and a segment table with a link to Churn Risk filtered by segment. Source: `/segments`.

### 4.5 Churn Risk (viewer+; export analyst+)
```
Band [All|High|Medium|Low]  Segment [All ▼]  Search customer key
[Counts: High 214 · Medium 531 · Low 1,095]  [Revenue at risk £48k]
+-------------------------------+  +-----------------------------------------+
| Probability histogram with    |  | Selected customer: reasons (horizontal   |
| band thresholds               |  | bars, plain language) + RFM + segment    |
+-------------------------------+  +-----------------------------------------+
| Table: customer · segment · probability · band · top reason   [Export CSV]   |
```
Sources: `/churn/summary`, `/customers`, `/customers/{id}`, `/customers/export`.

Reason wording (feature → plain language):
| Feature | Text |
|---------|------|
| `overdue_ratio` | Has gone longer than usual without ordering |
| `recency_days` | Long time since last order |
| `orders_last_90d` | Fewer orders recently |
| `order_trend` | Ordering less often than before |
| `avg_order_value` | Smaller orders than usual |
| `return_rate` | High share of items returned |
| `tenure_days` | Relationship length |
| `n_distinct_skus` | Narrow range of products bought |

Unknown features fall back to a humanized feature name.

### 4.6 Inventory (viewer+; edit and simulate analyst+)
```
Status [Reorder now ▼]  ABC [All ▼]  Search SKU            [Synthetic parameters: 100%] ⓘ
+--------------------------------------------------------------------------------------+
| SKU · description · ABC · on hand · on order · reorder point · rec. qty · risk bar · cover · status |
+--------------------------------------------------------------------------------------+
| Selected SKU: forecast chart over lead time + ROP/stock lines | What-if form [Save] |
+--------------------------------------------------------------------------------------+
| Policy simulation: fill rate · stockout days · avg inventory (RetailPulse vs baseline)  [Run] |
```
- `stockout_risk` shown as a progress bar plus percentage text. Status badges carry icon + text.
- A persistent notice appears whenever `params_source = synthetic`: "Lead times, costs and stock levels are simulated. Results show the method, not real savings."
- What-if edits recompute via `/inventory/what-if` and show the delta; **Save** calls `PUT /inventory/params/{id}` after a confirmation.
- Export button (analyst+) downloads `/inventory/recommendations/export`.

### 4.7 Model Monitoring (viewer+; actions admin)
Tabs: **Models** (runs table with algorithm, WAPE, MASE, bias, coverage, gate result, champion badge; compare up to 4; Promote and Train buttons for admin), **Performance** (realized vs backtest vs baseline WAPE over time), **Drift** (reports table, drift-share timeline, open the stored HTML in an isolated viewer), **Alerts** (open alerts with Acknowledge for analyst+). Sources: `/models`, `/monitoring/*`, `/alerts`.

### 4.8 Data & Pipelines (viewer+; upload analyst+)
Upload panel (file picker, source selector, size and type hints), batches table (status chips, rows loaded/rejected, link to quality results), pipeline runs table (flow, trigger, status, duration, error), Trigger buttons (`etl`, `score_all` for analyst+; others admin). Auto-refresh every 60 s while a run is queued or running. Sources: `/uploads`, `/pipelines/runs`, `/pipelines/{flow}/trigger`.

### 4.9 Admin: Users (admin)
Table of users (email, role, active, last login). Create user dialog, change role, deactivate. The last active admin cannot be demoted or deactivated (API enforces, UI explains).

## 5. Components

| Component | Behavior |
|-----------|----------|
| **KPI card** | `st.metric` with label, value, delta text, `help` tooltip defining the metric |
| **Status badge** | Icon + text + color: ok ✔, reorder now ⚠, overstock ◔, no data –, high/medium/low risk ● |
| **Filter bar** | Top of page; persists in `st.session_state` per page; changing a filter resets pagination |
| **Data table** | `st.dataframe` with `column_config`; server-side pagination and sorting via API; sticky header; numeric columns right-aligned; max 50 rows per page |
| **Chart** | Plotly figure built by helpers in `dashboard/components/charts.py`; always titled, axis-labelled with units, hover tooltips; followed by a one-line text summary (`st.caption`) |
| **Confirm dialog** | Promote, Train, Save params, Deactivate user: states the effect and offers Cancel; use `st.dialog` (or equivalent supported by the pinned version) |
| **Toast / message** | Success via `st.toast`; persistent errors inline |
| **Download button** | `st.download_button` with bytes from the API export endpoint; filename includes date |

Chart helper (shared styling):
```python
def style_figure(fig, *, title=None, y_title=None):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        margin=dict(l=48, r=16, t=48 if title else 16, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(zeroline=False)
    return fig
```

## 6. States

| State | Pattern |
|-------|---------|
| Loading | `st.spinner` with specific text ("Loading forecast…"); avoid full-page blocking |
| Empty | Icon-free message + the next action ("No forecast yet. An admin can run scoring from Data & Pipelines.") |
| Error | Inline error with plain message, **request ID**, and a Retry button; never show raw exception text or stack traces |
| Forbidden (403) | "You don't have access to this action." Hide the control next time (role-based UI) |
| Session expired (401) | Clear session, return to Login with "Your session expired. Please sign in again." |
| Stale data | Warning banner in status strip when the last successful scoring is older than 36 h |
| Partial data | Show what loaded; failed widgets show their own error cards |

## 7. Streamlit implementation notes

- `st.set_page_config(page_title="RetailPulse", layout="wide")` must be the first Streamlit call in `app.py`.
- Auth state lives in `st.session_state["auth"]` (`token`, `role`, `email`, `expires_at`). The token is never written to query params, logs or the page.
- `client.py` wraps `httpx`: connect timeout 3 s, read timeout 15 s, GET retries ≤ 2, maps problem-details responses to `ApiError(status, title, detail, request_id)`.
- **Caching:** `@st.cache_data(ttl=60)` on GET helpers; the cache key must include `role`; never cache `/auth/*`, `/users*`, admin-only or user-specific responses; clear affected caches after any mutation.
- **Never** pass API-provided strings to `unsafe_allow_html=True`. Prefer native components for badges and layout.
- Use `st.fragment(run_every=...)` (if supported by the pinned Streamlit) for auto-refresh sections instead of reloading the page.
- For `st.plotly_chart`, use the width parameter supported by the pinned version (`use_container_width=True` or `width="stretch"`).
- Each view module exposes `render()` and contains **no business logic**: fetch, shape, display.
- Keep view files small (< 250 lines); shared widgets go in `components/`.

## 8. Accessibility checklist

- [ ] Text contrast ≥ 4.5:1 (use `*_text` colors); chart colors from the color-blind-safe palette.
- [ ] Status and risk are never conveyed by color alone (icon and text).
- [ ] Every chart has a title, axis labels with units and a text summary.
- [ ] Tables have clear column headers; no information only in hover tooltips.
- [ ] All actions reachable by keyboard (rely on native Streamlit widgets).
- [ ] No auto-playing motion; auto-refresh limited to the Overview and Pipelines pages and can be switched off.
- [ ] Plain-language microcopy; metric tooltips define WAPE, MASE, coverage, safety stock, reorder point.

## 9. Microcopy

Use sentence case, short verbs ("Promote model", "Run simulation"). Avoid jargon without a tooltip. Dates in UI use `DD Mon YYYY`. Always name the unit ("units", "£"). Error text: what happened + what to do + request ID.
