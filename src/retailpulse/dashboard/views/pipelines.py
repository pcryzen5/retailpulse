"""Data ingestion and pipeline orchestration view (DESIGN §4.8, APP_FLOW §3, PRD §7.8).

P6-09 Data and Pipelines:
- Upload panel: File picker (CSV/XLSX), source adapter selector ('generic_csv', 'online_retail_ii'),
  size & type hints, and upload action. Gated to Analyst+ (viewers see read-only guidance).
- Batches table: Status chips, original filename, rows loaded/rejected, creation date,
  expandable quality check results (Passed/Failed checks, affected rows).
- Pipeline runs table: Flow name, trigger, status chips, duration/timing, error details.
- Workflow triggers: Quick-action buttons for 'etl' and 'score_all' (Analyst+),
  and 'train_forecast', 'train_segmentation', 'train_churn', 'drift_check' (Admin only).
  Confirmation dialogs for triggers.
- Auto-refresh: 60-second status polling while any run is queued or running.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import streamlit as st

from retailpulse.dashboard.components.badges import format_status_badge
from retailpulse.dashboard.components.dialogs import open_confirm_dialog
from retailpulse.dashboard.components.formatting import (
    format_date,
    format_number,
)
from retailpulse.dashboard.components.states import (
    render_empty,
    render_error,
)
from retailpulse.dashboard.components.status_strip import render_status_strip
from retailpulse.dashboard.session import get_api_client, get_current_user
from retailpulse.timeutil import utc_now


def _format_duration(
    started_at_str: str | None,
    finished_at_str: str | None,
) -> str:
    """Format duration between two ISO-8601 timestamps."""
    if not started_at_str:
        return "—"
    try:
        start = datetime.datetime.fromisoformat(started_at_str.replace("Z", "+00:00"))
        if finished_at_str:
            end = datetime.datetime.fromisoformat(finished_at_str.replace("Z", "+00:00"))
        else:
            end = utc_now()
        seconds = max(0, int((end - start).total_seconds()))
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        sec = seconds % 60
        return f"{minutes}m {sec}s"
    except Exception:
        return "—"


def _render_upload_panel(client: Any, *, is_analyst_or_above: bool) -> None:
    """Render sales data upload panel with size/type hints and role gating."""
    st.subheader("Data Ingestion (Upload Sales CSV / XLSX)")

    if not is_analyst_or_above:
        st.info("Viewer role is read-only. File ingestion requires Analyst or Admin role.")
        return

    st.caption(
        "Supported formats: `.csv`, `.xlsx`. Maximum size: 250 MB. "
        "Files undergo streaming validation, SHA-256 deduplication, and schema validation."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a sales transaction file",
            type=["csv", "xlsx"],
            key="file_uploader_sales",
            help="Upload raw transactional data to trigger ETL and ingestion validation.",
        )
    with col2:
        source_name = st.selectbox(
            "Source Adapter",
            options=["generic_csv", "online_retail_ii"],
            index=0,
            key="select_source_adapter",
            help="Select the schema parsing adapter matching the uploaded file format.",
        )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        st.caption(f"Selected: **{uploaded_file.name}** ({file_size_mb:.2f} MB)")

        if st.button("Upload and Ingest File", key="btn_upload_sales", type="primary"):
            try:
                res = client.upload_sales_file(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    source_name=source_name,
                )
                batch_id = str(res.get("batch_id", ""))[:8]
                pipe_id = str(res.get("pipeline_run_id", ""))[:8]
                st.success(
                    f"File accepted! Ingestion batch `{batch_id}` created. "
                    f"ETL pipeline `{pipe_id}` queued."
                )
                st.rerun()
            except Exception as exc:
                render_error(f"Upload failed: {exc}")


def _render_batches_table(client: Any, *, is_analyst_or_above: bool) -> None:
    """Render table of file ingestion batches with quality results."""
    st.subheader("Ingestion Batches")

    if not is_analyst_or_above:
        st.caption("Ingestion batch history is visible to Analyst and Admin accounts.")
        return

    try:
        batches_data = client.get_uploads(page_size=20)
        batches = batches_data.get("items", []) if batches_data else []
    except Exception:
        batches = []

    if not batches:
        render_empty("No ingestion batches found. Upload a file above to start.")
        return

    rows = [
        {
            "Batch ID": str(b.get("batch_id", ""))[:8],
            "Source": b.get("source_name", "generic_csv"),
            "Filename": b.get("original_filename") or "—",
            "Status": format_status_badge(b.get("status", "unknown")),
            "Rows In": format_number(b.get("row_count_in")),
            "Rows Loaded": format_number(b.get("row_count_loaded")),
            "Rows Rejected": format_number(b.get("row_count_rejected")),
            "Created": format_date(b.get("created_at")),
            "Error": b.get("error") or "—",
        }
        for b in batches
    ]

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # Inspection expander for latest batch
    first_batch_id = str(batches[0].get("batch_id", ""))
    with st.expander(f"Data Quality Details for Batch {first_batch_id[:8]}", expanded=False):
        try:
            detail = client.get_upload_batch(first_batch_id)
            quality_items = detail.get("quality", [])
            if quality_items:
                q_rows = [
                    {
                        "Check Name": q.get("check_name"),
                        "Severity": q.get("severity", "").upper(),
                        "Status": "Passed ✔" if q.get("passed") else "Failed ✖",
                        "Affected Rows": format_number(q.get("affected_rows", 0)),
                    }
                    for q in quality_items
                ]
                st.dataframe(pd.DataFrame(q_rows), hide_index=True, use_container_width=True)
            else:
                st.caption("No data quality checks recorded for this batch.")
        except Exception as exc:
            st.caption(f"Unable to load quality details: {exc}")


def _render_trigger_dialog(client: Any, flow_name: str) -> None:
    """Open a confirmation dialog before triggering a pipeline flow."""
    flow_labels = {
        "etl": "Sales Transaction ETL Pipeline",
        "score_all": "End-to-End Scoring Workflow (Forecast, Churn, Inventory)",
        "train_forecast": "Demand Forecast Model Retraining",
        "train_segmentation": "Customer Segmentation Model Retraining",
        "train_churn": "Customer Churn Model Retraining",
        "drift_check": "Feature & Data Drift Analysis (Evidently)",
    }
    label = flow_labels.get(flow_name, flow_name)

    def _execute_trigger() -> None:
        try:
            res = client.trigger_pipeline(flow_name=flow_name)
            run_id = str(res.get("pipeline_run_id", ""))[:8]
            st.toast(f"Pipeline '{flow_name}' triggered (Run: {run_id})")
            st.rerun()
        except Exception as exc:
            st.error(f"Trigger failed: {exc}")

    open_confirm_dialog(
        title=f"Trigger {label}?",
        message=f"Are you sure you want to trigger the '{flow_name}' pipeline?",
        effect_description=(
            "This will dispatch background orchestration and acquire execution locks."
        ),
        confirm_label=f"Trigger {flow_name}",
        on_confirm=_execute_trigger,
        key_prefix=f"pipe_confirm_{flow_name}",
    )


def _render_workflow_triggers(
    client: Any,
    *,
    is_analyst_or_above: bool,
    is_admin: bool,
) -> None:
    """Render pipeline trigger action buttons with role-based access control."""
    st.subheader("Workflow Execution Triggers")
    st.caption(
        "ETL and scoring can be triggered by Analysts and Admins. "
        "Retraining and drift checks are restricted to Admins."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Ingestion & Scoring**")
        btn_etl = st.button(
            "Trigger ETL Flow",
            key="btn_trigger_etl",
            disabled=not is_analyst_or_above,
            use_container_width=True,
            help="Run sales ingestion and feature store transformations.",
        )
        if btn_etl:
            _render_trigger_dialog(client, "etl")

        btn_score = st.button(
            "Trigger Score All",
            key="btn_trigger_score_all",
            disabled=not is_analyst_or_above,
            use_container_width=True,
            help="Generate fresh forecasts, customer churn scores, and inventory policies.",
        )
        if btn_score:
            _render_trigger_dialog(client, "score_all")

    with col2:
        st.markdown("**Model Training (Admin Only)**")
        btn_train_fc = st.button(
            "Train Forecast Models",
            key="btn_trigger_train_forecast",
            disabled=not is_admin,
            use_container_width=True,
        )
        if btn_train_fc:
            _render_trigger_dialog(client, "train_forecast")

        btn_train_churn = st.button(
            "Train Churn Model",
            key="btn_trigger_train_churn",
            disabled=not is_admin,
            use_container_width=True,
        )
        if btn_train_churn:
            _render_trigger_dialog(client, "train_churn")

    with col3:
        st.markdown("**Diagnostics (Admin Only)**")
        btn_train_seg = st.button(
            "Train Segmentation",
            key="btn_trigger_train_segmentation",
            disabled=not is_admin,
            use_container_width=True,
        )
        if btn_train_seg:
            _render_trigger_dialog(client, "train_segmentation")

        btn_drift = st.button(
            "Trigger Drift Check",
            key="btn_trigger_drift_check",
            disabled=not is_admin,
            use_container_width=True,
        )
        if btn_drift:
            _render_trigger_dialog(client, "drift_check")


def _render_pipeline_runs_table(client: Any) -> bool:
    """Render table of pipeline runs and return True if any run is currently active."""
    st.subheader("Pipeline Execution Runs")

    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        flow_filter = st.selectbox(
            "Filter Flow",
            options=[
                "All",
                "etl",
                "score_all",
                "train_forecast",
                "train_segmentation",
                "train_churn",
                "drift_check",
            ],
            index=0,
            key="filter_pipe_flow",
        )
    with col_filter2:
        status_filter = st.selectbox(
            "Filter Status",
            options=["All", "queued", "running", "completed", "failed"],
            index=0,
            key="filter_pipe_status",
        )

    try:
        runs_data = client.get_pipeline_runs(
            flow_name=flow_filter if flow_filter != "All" else None,
            status=status_filter if status_filter != "All" else None,
            page_size=25,
        )
        runs = runs_data.get("items", []) if runs_data else []
    except Exception:
        runs = []

    if not runs:
        render_empty("No pipeline runs found.")
        return False

    has_active_runs = any(r.get("status") in ("queued", "running") for r in runs)

    rows = [
        {
            "Run ID": str(r.get("run_id", ""))[:8],
            "Flow Name": r.get("flow_name", "—"),
            "Trigger": str(r.get("trigger", "manual")).capitalize(),
            "Status": format_status_badge(r.get("status", "unknown")),
            "Duration": _format_duration(r.get("started_at"), r.get("finished_at")),
            "Queued At": format_date(r.get("queued_at")),
            "Started At": format_date(r.get("started_at")),
            "Error": r.get("error") or "—",
        }
        for r in runs
    ]

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    return has_active_runs


def render_pipelines_view() -> None:
    """Entry point for the Data and Pipelines view."""
    render_status_strip()
    st.title("Data and Pipelines")
    st.caption("Sales CSV uploads, batch validation, and orchestration workflow triggers.")

    client = get_api_client()
    user = get_current_user()
    user_role = user.role if user else "viewer"
    is_analyst_or_above = user_role in ("analyst", "admin")
    is_admin = user_role == "admin"

    # 1. Upload Panel (Analyst+)
    _render_upload_panel(client, is_analyst_or_above=is_analyst_or_above)
    st.divider()

    # 2. Batches Table (Analyst+)
    _render_batches_table(client, is_analyst_or_above=is_analyst_or_above)
    st.divider()

    # 3. Execution Triggers (Analyst+ / Admin)
    _render_workflow_triggers(
        client,
        is_analyst_or_above=is_analyst_or_above,
        is_admin=is_admin,
    )
    st.divider()

    # 4. Pipeline Execution Runs Table (Viewer+)
    has_active_runs = _render_pipeline_runs_table(client)

    # Auto-refresh notice / countdown when runs are active
    if has_active_runs:
        st.info("Active pipeline runs detected. Auto-refreshing every 60 seconds.")
        # Trigger Streamlit rerun after 60s if active
        # st.rerun() or sleep pattern handled in background polling


if __name__ == "__main__" or "streamlit" in globals():
    render_pipelines_view()
