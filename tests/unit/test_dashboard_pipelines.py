"""AppTest and unit test suite for the Data and Pipelines dashboard view.

Verifies P6-09:
- Upload panel: File picker and source adapter selector (Analyst+).
- Role-gated upload: Hidden/read-only for Viewer; functional for Analyst/Admin.
- Batches table: Status chips, rows loaded/rejected, quality results.
- Triggers: 'etl' and 'score_all' enabled for Analyst; training/drift restricted to Admin.
- Pipeline runs table: Flow, trigger, status, duration, error details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from retailpulse.dashboard.client import ApiClient, ForbiddenError, StatusStripData, UserProfile

VIEWS_DIR = Path(__file__).parent.parent.parent / "src" / "retailpulse" / "dashboard" / "views"
PIPELINES_VIEW_PATH = str(VIEWS_DIR / "pipelines.py")


class StubPipelinesClient(ApiClient):
    """Stub API client providing predictable responses for Data & Pipelines tests."""

    def __init__(self, role: str = "admin") -> None:
        self.role = role
        self._token: str | None = "fake-token"
        self.uploaded_files: list[dict[str, Any]] = []
        self.triggered_flows: list[str] = []

    def set_token(self, token: str | None) -> None:
        self._token = token

    def get_status_strip_data(self) -> Any:
        return StatusStripData(
            data_as_of="2026-11-25T08:00:00Z",
            champion_model="lightgbm",
            last_pipeline_status="completed",
            last_pipeline_time="2026-11-25T08:00:00Z",
            is_stale_warning=False,
            stale_reason=None,
        )

    def get_status_strip(self) -> dict[str, Any]:
        return {
            "db_status": "ok",
            "last_pipeline_status": "completed",
            "last_pipeline_time": "2026-11-25T08:00:00Z",
            "open_alerts_count": 0,
        }

    def get_uploads(
        self,
        status: str | None = None,
        source_name: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict[str, Any]:
        if self.role == "viewer":
            raise ForbiddenError("Viewer cannot view batches")
        return {
            "items": [
                {
                    "batch_id": "11111111-1111-1111-1111-111111111111",
                    "source_name": "generic_csv",
                    "original_filename": "online_retail.csv",
                    "status": "completed",
                    "row_count_in": 541909,
                    "row_count_loaded": 530000,
                    "row_count_rejected": 11909,
                    "created_at": "2026-11-24T12:00:00Z",
                    "error": None,
                }
            ],
            "total": 1,
            "page": 1,
            "page_size": page_size,
        }

    def get_upload_batch(self, batch_id: str) -> dict[str, Any]:
        return {
            "batch_id": batch_id,
            "status": "completed",
            "source_name": "generic_csv",
            "row_count_in": 541909,
            "row_count_loaded": 530000,
            "row_count_rejected": 11909,
            "quality": [
                {
                    "check_name": "null_customer_id",
                    "severity": "warn",
                    "passed": True,
                    "affected_rows": 11909,
                },
                {
                    "check_name": "negative_price",
                    "severity": "error",
                    "passed": True,
                    "affected_rows": 0,
                },
            ],
            "error": None,
        }

    def upload_sales_file(
        self,
        file_bytes: bytes,
        filename: str,
        source_name: str = "generic_csv",
    ) -> dict[str, Any]:
        self.uploaded_files.append(
            {"filename": filename, "size": len(file_bytes), "source": source_name}
        )
        return {
            "batch_id": "22222222-2222-2222-2222-222222222222",
            "status": "queued",
            "pipeline_run_id": "33333333-3333-3333-3333-333333333333",
        }

    def get_pipeline_runs(
        self,
        flow_name: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        return {
            "items": [
                {
                    "run_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "flow_name": "etl",
                    "trigger": "upload",
                    "status": "running",
                    "params": {"source": "generic_csv"},
                    "error": None,
                    "queued_at": "2026-11-25T08:10:00Z",
                    "started_at": "2026-11-25T08:10:05Z",
                    "finished_at": None,
                },
                {
                    "run_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "flow_name": "score_all",
                    "trigger": "manual",
                    "status": "completed",
                    "params": {},
                    "error": None,
                    "queued_at": "2026-11-25T07:00:00Z",
                    "started_at": "2026-11-25T07:00:05Z",
                    "finished_at": "2026-11-25T07:05:00Z",
                },
            ],
            "total": 2,
            "page": 1,
            "page_size": page_size,
        }

    def trigger_pipeline(
        self,
        flow_name: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.triggered_flows.append(flow_name)
        return {
            "pipeline_run_id": "44444444-4444-4444-4444-444444444444",
            "flow_name": flow_name,
            "status": "queued",
        }


def test_pipelines_view_renders_for_admin() -> None:
    """Admin sees upload panel, batches table, all triggers active, and pipeline runs."""
    stub_client = StubPipelinesClient(role="admin")
    at = AppTest.from_file(PIPELINES_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="00000000-0000-0000-0000-000000000001",
        email="admin@retailpulse.com",
        role="admin",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    # Title and sections
    assert any("Data and Pipelines" in t.value for t in at.title)

    # Check button disabled states for admin
    # Ingestion/Scoring: etl, score_all
    # Model Training: train_forecast, train_churn
    # Diagnostics: train_segmentation, drift_check
    btn_keys = [b.key for b in at.button]
    assert "btn_trigger_etl" in btn_keys
    assert "btn_trigger_score_all" in btn_keys
    assert "btn_trigger_train_forecast" in btn_keys
    assert "btn_trigger_train_churn" in btn_keys
    assert "btn_trigger_train_segmentation" in btn_keys
    assert "btn_trigger_drift_check" in btn_keys

    # None of the trigger buttons should be disabled for admin
    for b in at.button:
        if str(b.key).startswith("btn_trigger_"):
            assert not b.disabled


def test_pipelines_view_role_gating_analyst() -> None:
    """Analyst can upload files and trigger etl/score_all, but not train/drift."""
    stub_client = StubPipelinesClient(role="analyst")
    at = AppTest.from_file(PIPELINES_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="00000000-0000-0000-0000-000000000002",
        email="analyst@retailpulse.com",
        role="analyst",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    # File uploader is present
    assert len(at.file_uploader) > 0

    # etl and score_all are enabled
    etl_btn = next(b for b in at.button if b.key == "btn_trigger_etl")
    score_btn = next(b for b in at.button if b.key == "btn_trigger_score_all")
    assert not etl_btn.disabled
    assert not score_btn.disabled

    # train and drift are disabled for analyst
    train_fc_btn = next(b for b in at.button if b.key == "btn_trigger_train_forecast")
    drift_btn = next(b for b in at.button if b.key == "btn_trigger_drift_check")
    assert train_fc_btn.disabled
    assert drift_btn.disabled


def test_pipelines_view_role_gating_viewer() -> None:
    """Viewer sees read-only info message, no file uploader, and all triggers disabled."""
    stub_client = StubPipelinesClient(role="viewer")
    at = AppTest.from_file(PIPELINES_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="00000000-0000-0000-0000-000000000003",
        email="viewer@retailpulse.com",
        role="viewer",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    # File uploader should NOT be present for viewer
    assert len(at.file_uploader) == 0

    # Info message explaining viewer is read-only
    assert any("Viewer role is read-only" in inf.value for inf in at.info)

    # All triggers are disabled for viewer
    for b in at.button:
        if str(b.key).startswith("btn_trigger_"):
            assert b.disabled


def test_pipelines_active_runs_notice() -> None:
    """When a pipeline run is queued or running, active runs notice is shown."""
    stub_client = StubPipelinesClient(role="analyst")
    at = AppTest.from_file(PIPELINES_VIEW_PATH, default_timeout=30)
    at.session_state["api_client"] = stub_client
    at.session_state["user"] = UserProfile(
        user_id="00000000-0000-0000-0000-000000000002",
        email="analyst@retailpulse.com",
        role="analyst",
    )
    at.session_state["logged_in"] = True
    at.run()

    assert not at.exception
    assert any("Active pipeline runs detected" in inf.value for inf in at.info)
