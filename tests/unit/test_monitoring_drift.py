"""Contract tests for Evidently drift monitoring wrapper.

Implements P7-04 verification per ARCHITECTURE §7 and TRD §6:
- Contract test of the wrapper on the test fixture (`tests/fixtures/mini_retail.csv`).
- Verifies feature (data) drift detection and prediction drift detection.
- Verifies HTML report generation and file persistence.
- Verifies summary metrics extraction: dataset_drift (bool), share_drifted_features (float/Decimal),
  n_features, n_drifted, per-feature p-values and stat tests.
- Verifies database persistence to `ops.drift_report` with schema check constraints.
- Verifies API endpoint compatibility for serving stored reports with sandbox CSP headers.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from starlette.testclient import TestClient

from retailpulse.api.app import create_app
from retailpulse.db.base import Base
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.ops import DriftReport
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError
from retailpulse.etl.clean import clean_sales_lines
from retailpulse.ingestion.adapters.online_retail_ii import OnlineRetailIIAdapter
from retailpulse.monitoring.drift import (
    DriftAnalysisResult,
    compute_drift_report,
    compute_forecast_drift,
    run_drift_analysis,
    save_drift_report,
)
from retailpulse.security.jwt import create_access_token
from retailpulse.security.password import hash_password
from retailpulse.timeutil import utc_now


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def db_engine():
    """In-memory SQLite engine with translated schemas."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture
def db_session(db_engine: sa.Engine):
    """Database session connected to the test SQLite engine."""
    session_cls = sessionmaker(bind=db_engine)
    session = session_cls()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def fixture_dataframe() -> pd.DataFrame:
    """Load cleaned transaction data from tests/fixtures/mini_retail.csv."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    if not fixture_path.exists():
        pytest.skip("mini_retail.csv fixture not found")

    raw_df = OnlineRetailIIAdapter.read_csv(fixture_path)
    clean_result = clean_sales_lines(raw_df, salt=b"test-salt-secret-0123456789abcdef")
    df = clean_result.df.copy()

    # Extract numeric and analytical features
    df["revenue"] = df["quantity"] * df["unit_price"]
    return df


# ==============================================================================
# 1. Contract Tests on Fixture Data
# ==============================================================================


def test_evidently_wrapper_contract_on_fixture(fixture_dataframe: pd.DataFrame) -> None:
    """Contract test: Evidently wrapper produces valid summary and HTML on mini_retail fixture."""
    df = fixture_dataframe.sort_values("invoice_ts").reset_index(drop=True)
    assert len(df) >= 20, "Fixture must contain sufficient transactions for drift evaluation"

    # Split into temporal reference window (first half) and current window (second half)
    half = len(df) // 2
    ref_df = df.iloc[:half][["quantity", "unit_price", "revenue"]].copy()
    curr_df = df.iloc[half:][["quantity", "unit_price", "revenue"]].copy()

    # Execute Evidently wrapper
    result = run_drift_analysis(
        reference_data=ref_df,
        current_data=curr_df,
        task="forecast",
        kind="data",
    )

    # Verify DriftAnalysisResult contract fields
    assert isinstance(result, DriftAnalysisResult)
    assert result.task == "forecast"
    assert result.kind == "data"
    assert isinstance(result.dataset_drift, bool)
    assert isinstance(result.share_drifted_features, float)
    assert 0.0 <= result.share_drifted_features <= 1.0

    assert result.n_features == 3
    assert result.n_drifted_features == len(result.drifted_features)
    assert set(result.drift_by_feature.keys()) == {"quantity", "unit_price", "revenue"}

    # Verify per-feature contract
    for _feat_name, feat_meta in result.drift_by_feature.items():
        assert "drift_detected" in feat_meta
        assert isinstance(feat_meta["drift_detected"], bool)
        assert "score" in feat_meta
        assert isinstance(feat_meta["score"], float)
        assert "threshold" in feat_meta
        assert "method" in feat_meta

    # Verify summary dict contract
    summary = result.summary
    assert summary["task"] == "forecast"
    assert summary["kind"] == "data"
    assert summary["n_features"] == 3
    assert "n_drifted" in summary
    assert "share_drifted_features" in summary
    assert "dataset_drift" in summary
    assert "drifted_features" in summary

    # Verify HTML report contract
    assert isinstance(result.html_report, str)
    assert len(result.html_report) > 500
    assert ("<!DOCTYPE html>" in result.html_report or "<html" in result.html_report)


def test_evidently_wrapper_detects_data_drift(fixture_dataframe: pd.DataFrame) -> None:
    """Contract test: wrapper correctly flags drift when feature distributions are shifted."""
    df = fixture_dataframe[["quantity", "unit_price", "revenue"]].copy()
    ref_df = df.copy()

    # Force artificial distribution drift on quantity and revenue
    curr_df = df.copy()
    curr_df["quantity"] = curr_df["quantity"] * 25.0 + 500.0
    curr_df["revenue"] = curr_df["revenue"] * 30.0 + 2000.0

    result = run_drift_analysis(
        reference_data=ref_df,
        current_data=curr_df,
        task="forecast",
        kind="data",
        drift_share_threshold=0.5,
    )

    # 2 out of 3 features drifted (66.7% > 50% threshold)
    assert result.dataset_drift is True
    assert result.n_drifted_features >= 2
    assert "quantity" in result.drifted_features
    assert "revenue" in result.drifted_features
    assert result.share_drifted_features >= 0.5
    assert result.drift_by_feature["quantity"]["drift_detected"] is True


def test_evidently_wrapper_prediction_drift() -> None:
    """Contract test: wrapper evaluates prediction drift on model output series."""
    rng = np.random.default_rng(seed=42)

    # Baseline reference predictions: Normal(mean=50, std=10)
    ref_preds = pd.DataFrame({"prediction": rng.normal(loc=50.0, scale=10.0, size=200)})

    # Current predictions with slight, non-drift variation
    curr_similar = pd.DataFrame({"prediction": rng.normal(loc=50.2, scale=9.8, size=200)})

    res_no_drift = run_drift_analysis(
        reference_data=ref_preds,
        current_data=curr_similar,
        task="forecast",
        kind="prediction",
    )
    assert res_no_drift.kind == "prediction"
    assert res_no_drift.n_features == 1
    assert res_no_drift.dataset_drift is False
    assert res_no_drift.n_drifted_features == 0

    # Current predictions with severe mean shift: Normal(mean=95, std=15)
    curr_shifted = pd.DataFrame({"prediction": rng.normal(loc=95.0, scale=15.0, size=200)})

    res_drift = run_drift_analysis(
        reference_data=ref_preds,
        current_data=curr_shifted,
        task="forecast",
        kind="prediction",
    )
    assert res_drift.kind == "prediction"
    assert res_drift.dataset_drift is True
    assert res_drift.n_drifted_features == 1
    assert "prediction" in res_drift.drifted_features


# ==============================================================================
# 2. Database Persistence and File Storage Contract
# ==============================================================================


def test_save_drift_report_database_and_html_file(
    db_session: Session,
    tmp_path: Path,
) -> None:
    """Verify save_drift_report writes HTML to disk and commits summary row to DB."""
    ref_df = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0, 5.0], "f2": [10.0, 20.0, 30.0, 40.0, 50.0]})
    curr_df = pd.DataFrame({"f1": [1.1, 2.1, 2.9, 4.2, 5.1], "f2": [11.0, 19.0, 31.0, 39.0, 52.0]})

    analysis = run_drift_analysis(
        reference_data=ref_df,
        current_data=curr_df,
        task="forecast",
        kind="data",
    )

    today = utc_now().date()
    model_run_id = uuid.uuid4()

    drift_report = save_drift_report(
        result=analysis,
        session=db_session,
        model_run_id=model_run_id,
        reference_start=today - datetime.timedelta(days=60),
        reference_end=today - datetime.timedelta(days=30),
        current_start=today - datetime.timedelta(days=29),
        current_end=today,
        reports_dir=tmp_path,
    )

    # 1. Assert file exists on disk with HTML content
    expected_file = tmp_path / drift_report.report_path
    assert expected_file.exists()
    assert expected_file.is_file()
    assert expected_file.read_text(encoding="utf-8") == analysis.html_report

    # 2. Assert database record fields
    assert isinstance(drift_report.report_id, uuid.UUID)
    assert drift_report.task == "forecast"
    assert drift_report.kind == "data"
    assert drift_report.model_run_id == model_run_id
    assert drift_report.dataset_drift is False
    assert isinstance(drift_report.share_drifted_features, Decimal)
    assert drift_report.summary["n_features"] == 2
    assert "html_content" in drift_report.summary

    # Query back from DB
    queried = db_session.get(DriftReport, drift_report.report_id)
    assert queried is not None
    assert queried.report_path == drift_report.report_path


def test_compute_drift_report_convenience_pipeline(
    db_session: Session,
    tmp_path: Path,
) -> None:
    """Verify compute_drift_report executes analysis and persists in one call."""
    ref_df = pd.DataFrame({"churn_score": [0.1, 0.2, 0.15, 0.05, 0.3]})
    curr_df = pd.DataFrame({"churn_score": [0.8, 0.9, 0.85, 0.95, 0.7]})

    report = compute_drift_report(
        reference_data=ref_df,
        current_data=curr_df,
        session=db_session,
        task="churn",
        kind="prediction",
        reports_dir=tmp_path,
    )

    assert report.task == "churn"
    assert report.kind == "prediction"
    assert report.dataset_drift is True
    assert (tmp_path / report.report_path).exists()


def test_compute_forecast_drift_query_fallback(
    db_session: Session,
    tmp_path: Path,
) -> None:
    """Verify compute_forecast_drift falls back gracefully when facts table is unpopulated."""
    report = compute_forecast_drift(
        session=db_session,
        kind="data",
        reports_dir=tmp_path,
    )
    assert report.task == "forecast"
    assert report.kind == "data"
    assert report.summary["n_features"] >= 2
    assert (tmp_path / report.report_path).exists()


# ==============================================================================
# 3. Input Validation and Domain Error Handling
# ==============================================================================


def test_drift_analysis_validation_errors() -> None:
    """Verify DomainValidationError on invalid inputs or mismatched schemas."""
    empty_df = pd.DataFrame()
    valid_df = pd.DataFrame({"a": [1, 2, 3]})

    # Empty reference
    with pytest.raises(DomainValidationError, match="Reference dataset cannot be empty"):
        run_drift_analysis(reference_data=empty_df, current_data=valid_df)

    # Empty current
    with pytest.raises(DomainValidationError, match="Current dataset cannot be empty"):
        run_drift_analysis(reference_data=valid_df, current_data=empty_df)

    # Invalid task
    with pytest.raises(DomainValidationError, match="Invalid task"):
        run_drift_analysis(reference_data=valid_df, current_data=valid_df, task="unsupported_task")

    # Invalid kind
    with pytest.raises(DomainValidationError, match="Invalid drift kind"):
        run_drift_analysis(reference_data=valid_df, current_data=valid_df, kind="unsupported_kind")

    # No shared columns
    other_df = pd.DataFrame({"x": [4, 5, 6]})
    with pytest.raises(DomainValidationError, match="No shared evaluation columns found"):
        run_drift_analysis(reference_data=valid_df, current_data=other_df)


# ==============================================================================
# 4. API Endpoint Integration Contract
# ==============================================================================


def test_api_serves_generated_drift_report_and_html(
    db_engine: sa.Engine,
    tmp_path: Path,
) -> None:
    """Verify API router /monitoring/drift and detail/html endpoints serve Evidently outputs."""
    app = create_app()

    def override_get_db():
        session_cls = sessionmaker(bind=db_engine)
        session = session_cls()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db

    session_cls = sessionmaker(bind=db_engine)
    session = session_cls()

    # Create viewer user
    viewer_id = uuid.uuid4()
    viewer = AppUser(
        user_id=viewer_id,
        email="viewer_drift@retailpulse.local",
        password_hash=hash_password("ViewerPassword123!"),
        role="viewer",
        is_active=True,
    )
    session.add(viewer)
    session.commit()

    # Generate and persist drift report using reports_dir = tmp_path
    ref_df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    curr_df = pd.DataFrame({
        "feat_a": [1.2, 1.9, 3.1, 3.9, 5.2],
        "feat_b": [9.0, 21.0, 29.0, 41.0, 49.0],
    })

    report = compute_drift_report(
        reference_data=ref_df,
        current_data=curr_df,
        session=session,
        task="forecast",
        kind="data",
        reports_dir=tmp_path,
    )
    session.close()

    viewer_token = create_access_token(user_id=viewer_id, role="viewer")
    client = TestClient(app)
    headers = {"Authorization": f"Bearer {viewer_token}"}

    # 1. GET /api/v1/monitoring/drift
    res_list = client.get("/api/v1/monitoring/drift", headers=headers)
    assert res_list.status_code == 200
    list_body = res_list.json()
    assert list_body["total"] >= 1
    found = [r for r in list_body["items"] if r["report_id"] == str(report.report_id)]
    assert len(found) == 1
    assert found[0]["task"] == "forecast"

    # 2. GET /api/v1/monitoring/drift/{report_id}
    res_detail = client.get(f"/api/v1/monitoring/drift/{report.report_id}", headers=headers)
    assert res_detail.status_code == 200
    detail_body = res_detail.json()
    assert detail_body["report_id"] == str(report.report_id)
    assert detail_body["dataset_drift"] is False
    assert "n_features" in detail_body["summary"]

    # 3. GET /api/v1/monitoring/drift/{report_id}/html with sandbox CSP headers
    # Note: report.summary contains fallback guaranteeing serving across path overrides
    res_html = client.get(f"/api/v1/monitoring/drift/{report.report_id}/html", headers=headers)
    assert res_html.status_code == 200
    assert "text/html" in res_html.headers.get("content-type", "")
    assert res_html.headers.get("Content-Security-Policy") == "sandbox"
    assert res_html.headers.get("X-Content-Type-Options") == "nosniff"
    assert len(res_html.text) > 500

    app.dependency_overrides.clear()
