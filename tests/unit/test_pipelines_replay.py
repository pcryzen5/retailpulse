"""Unit tests for historical replay mode and scheduled flow execution.

Verifies P7-07:
- `start_replay` holds back specified days (default: 60) and partitions daily slices.
- `step_replay` executes scheduled daily flows (etl, score_all, drift_check) and weekly retraining.
- Recovers gracefully from flow failures, captures errors in audit history, and continues.
- `generate_replay_report` and `run_replay_simulation` produce reports showing >= 95% success.
- CLI commands (`start`, `step`, `status`, `run`) work end-to-end.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.pipelines.replay import (
    ReplayState,
    generate_replay_report,
    get_replay_status,
    run_replay_simulation,
    start_replay,
    step_replay,
)

runner = CliRunner()


@pytest.fixture
def sample_sales_data(tmp_path: Path) -> Path:
    """Generate a synthetic sales dataset spanning 80 days for replay testing."""
    records = []
    base_date = datetime.date(2011, 9, 1)
    for day_offset in range(80):
        current_date = base_date + datetime.timedelta(days=day_offset)
        # Create 3 transactions per day
        for idx in range(3):
            records.append(
                {
                    "invoice_no": f"INV-{day_offset:03d}-{idx}",
                    "invoice_ts": f"{current_date.isoformat()} 10:00:00",
                    "sku": f"SKU-{idx:02d}",
                    "description": f"Product {idx}",
                    "quantity": 10 + idx,
                    "unit_price": 5.5,
                    "customer_id": f"1000{idx}",
                    "country": "United Kingdom",
                    "store_code": "ONLINE",
                }
            )

    df = pd.DataFrame(records)
    target_parquet = tmp_path / "synthetic_retail.parquet"
    df.to_parquet(target_parquet, index=False)
    return target_parquet


def test_start_replay_creates_partitions_and_state(
    sample_sales_data: Path,
    tmp_path: Path,
) -> None:
    """Verify start_replay calculates anchor date and creates partitioned day files."""
    staging_dir = tmp_path / "replay_staging"
    holdback_days = 20

    state = start_replay(
        holdback_days=holdback_days,
        source_path=sample_sales_data,
        staging_dir=staging_dir,
    )

    assert state.is_active is True
    assert state.holdback_days == 20
    assert state.current_step_index == 0
    assert len(state.staged_dates) == holdback_days

    # Verify anchor date calculation: max_date - 20 days
    max_dt = datetime.date.fromisoformat(state.max_date)
    anchor_dt = datetime.date.fromisoformat(state.anchor_date)
    assert anchor_dt == max_dt - datetime.timedelta(days=20)

    # Verify staged day partition files
    days_dir = staging_dir / "days"
    assert days_dir.is_dir()
    for d_str in state.staged_dates:
        day_file = days_dir / f"day_{d_str}.parquet"
        assert day_file.is_file(), f"Expected partition file for {d_str}"


def test_start_replay_invalid_holdback_days(sample_sales_data: Path, tmp_path: Path) -> None:
    """start_replay raises ValueError when holdback_days < 1."""
    staging_dir = tmp_path / "replay_staging"
    with pytest.raises(ValueError, match="holdback_days must be >= 1"):
        start_replay(holdback_days=0, source_path=sample_sales_data, staging_dir=staging_dir)


def test_step_replay_executes_daily_and_weekly_flows(
    sample_sales_data: Path,
    tmp_path: Path,
) -> None:
    """step_replay executes etl, score_all, drift_check daily and retraining weekly."""
    staging_dir = tmp_path / "replay_staging"
    start_replay(holdback_days=10, source_path=sample_sales_data, staging_dir=staging_dir)

    executed_flows: list[str] = []

    def mock_executor(flow_name: str, params: dict[str, Any]) -> dict[str, Any]:
        executed_flows.append(flow_name)
        if flow_name == "etl":
            assert "file_path" in params
            assert Path(params["file_path"]).is_file()
        return {"status": "ok"}

    # Advance by 1 day
    records_day1 = step_replay(days=1, staging_dir=staging_dir, flow_executor=mock_executor)
    assert len(records_day1) == 3
    day1_flows = [r["flow_name"] for r in records_day1]
    assert day1_flows == ["etl", "score_all", "drift_check"]
    assert all(r["status"] == "succeeded" for r in records_day1)

    # Advance by 6 more days (reaching step 7: triggers weekly retraining)
    records_next = step_replay(days=6, staging_dir=staging_dir, flow_executor=mock_executor)
    assert len(records_next) > 0

    # Inspect step 7
    step7_records = [r for r in records_next if r["step_index"] == 7]
    step7_flow_names = [r["flow_name"] for r in step7_records]
    assert "etl" in step7_flow_names
    assert "score_all" in step7_flow_names
    assert "drift_check" in step7_flow_names
    assert "train_forecast" in step7_flow_names
    assert "train_segmentation" in step7_flow_names
    assert "train_churn" in step7_flow_names


def test_step_replay_records_failure_and_continues(
    sample_sales_data: Path,
    tmp_path: Path,
) -> None:
    """When a flow fails during replay, the error is recorded and simulation continues."""
    staging_dir = tmp_path / "replay_staging"
    start_replay(holdback_days=5, source_path=sample_sales_data, staging_dir=staging_dir)

    def mock_flaky_executor(flow_name: str, params: dict[str, Any]) -> dict[str, Any]:
        if flow_name == "drift_check":
            raise RuntimeError("Evidently metric computation error")
        return {"status": "ok"}

    records = step_replay(days=1, staging_dir=staging_dir, flow_executor=mock_flaky_executor)
    assert len(records) == 3

    etl_rec = next(r for r in records if r["flow_name"] == "etl")
    assert etl_rec["status"] == "succeeded"

    drift_rec = next(r for r in records if r["flow_name"] == "drift_check")
    assert drift_rec["status"] == "failed"
    assert "Evidently metric computation error" in str(drift_rec["error"])

    # Verify state reflects failure
    state = ReplayState.load(staging_dir / "state.json")
    assert state.succeeded_flows_count == 2
    assert state.failed_flows_count == 1
    assert state.success_rate == pytest.approx(66.67, rel=1e-2)


def test_get_replay_status(sample_sales_data: Path, tmp_path: Path) -> None:
    """get_replay_status reflects progress before and after stepping."""
    staging_dir = tmp_path / "replay_staging"

    # Inactive
    inactive_status = get_replay_status(staging_dir=staging_dir)
    assert inactive_status["is_active"] is False

    # Active after start
    start_replay(holdback_days=10, source_path=sample_sales_data, staging_dir=staging_dir)
    status_active = get_replay_status(staging_dir=staging_dir)
    assert status_active["is_active"] is True
    assert status_active["holdback_days"] == 10
    assert status_active["current_step_index"] == 0
    assert status_active["days_remaining"] == 10


def test_generate_replay_report(sample_sales_data: Path, tmp_path: Path) -> None:
    """generate_replay_report outputs comprehensive Markdown with verification status."""
    staging_dir = tmp_path / "replay_staging"
    report_file = tmp_path / "test_report.md"

    start_replay(holdback_days=3, source_path=sample_sales_data, staging_dir=staging_dir)
    step_replay(
        days=2,
        staging_dir=staging_dir,
        flow_executor=lambda fn, p: {"status": "ok"},
    )

    state = ReplayState.load(staging_dir / "state.json")
    out_path = generate_replay_report(state, output_path=report_file)

    assert out_path.is_file()
    content = out_path.read_text(encoding="utf-8")
    assert "# RetailPulse — Operational Replay & Pipeline Reliability Report" in content
    assert "**Operational Verification Status**: **PASSED**" in content
    assert "100.00%" in content
    assert "Pipeline Flow Execution Summary" in content
    assert "`etl`" in content
    assert "`score_all`" in content
    assert "`drift_check`" in content
    assert "Verification Checklist & Compliance" in content


def test_run_replay_simulation_end_to_end(sample_sales_data: Path, tmp_path: Path) -> None:
    """run_replay_simulation runs full 14-day pipeline simulation and verifies >= 95% success."""
    staging_dir = tmp_path / "replay_staging"
    report_file = tmp_path / "replay_run.md"

    summary = run_replay_simulation(
        days=14,
        holdback_days=30,
        source_path=sample_sales_data,
        report_path=report_file,
        staging_dir=staging_dir,
        flow_executor=lambda fn, p: {"status": "ok"},
    )

    assert summary["status"] == "passed"
    assert summary["days_simulated"] == 14
    assert summary["success_rate"] >= 95.0
    assert summary["total_flows"] >= 42  # 14 * 3 daily + 2 weekly rounds
    assert report_file.is_file()


def test_cli_replay_commands(sample_sales_data: Path, tmp_path: Path) -> None:
    """Test CLI commands: start, status, step, and run."""
    staging_dir = tmp_path / "cli_replay"
    report_path = tmp_path / "cli_report.md"

    # 1. start
    res_start = runner.invoke(
        app,
        [
            "replay",
            "start",
            "--holdback-days",
            "15",
            "--source-path",
            str(sample_sales_data),
            "--staging-dir",
            str(staging_dir),
        ],
    )
    assert res_start.exit_code == 0
    assert "Successfully initialized replay mode" in res_start.output

    # 2. status
    res_status = runner.invoke(
        app,
        ["replay", "status", "--staging-dir", str(staging_dir)],
    )
    assert res_status.exit_code == 0
    assert "Replay Mode Status" in res_status.output
    assert "Holdback window:        15 calendar days" in res_status.output

    # JSON status
    res_json = runner.invoke(
        app,
        ["replay", "status", "--staging-dir", str(staging_dir), "--json"],
    )
    assert res_json.exit_code == 0
    assert '"is_active": true' in res_json.output

    # 3. step
    res_step = runner.invoke(
        app,
        ["replay", "step", "--days", "1", "--staging-dir", str(staging_dir)],
    )
    assert res_step.exit_code == 0
    assert "Advanced replay by" in res_step.output

    # 4. run
    res_run = runner.invoke(
        app,
        [
            "replay",
            "run",
            "--days",
            "3",
            "--holdback-days",
            "5",
            "--source-path",
            str(sample_sales_data),
            "--report-path",
            str(report_path),
            "--staging-dir",
            str(staging_dir),
        ],
    )
    assert res_run.exit_code == 0
    assert "Replay simulation finished successfully" in res_run.output
    assert report_path.is_file()
