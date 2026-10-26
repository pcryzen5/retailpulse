"""Unit tests for the RetailPulse CLI."""

from __future__ import annotations

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from retailpulse import __version__
from retailpulse.cli import app
from retailpulse.db.models.core import (
    DimCustomer,
    DimDate,
    DimProduct,
    DimStore,
    FactDailySales,
    FactSalesLine,
)
from retailpulse.db.models.ops import DataQualityResult, IngestionBatch

pytestmark = pytest.mark.unit

runner = CliRunner()


def test_cli_help() -> None:
    """Verify retailpulse --help exits with 0 and prints help text."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "RetailPulse" in result.output
    assert "version" in result.output


def test_cli_version_command() -> None:
    """Verify retailpulse version prints version string."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_version_flag() -> None:
    """Verify retailpulse --version prints version string."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_etl_help() -> None:
    """Verify retailpulse etl --help displays commands."""
    result = runner.invoke(app, ["etl", "--help"])
    assert result.exit_code == 0
    assert "run" in result.output


def test_etl_run_help() -> None:
    """Verify retailpulse etl run --help displays parameters."""
    result = runner.invoke(app, ["etl", "run", "--help"])
    assert result.exit_code == 0
    assert "--source" in result.output
    assert "--path" in result.output
    assert "--chunk-size" in result.output


def test_etl_run_missing_path() -> None:
    """Verify etl run without required --path option fails."""
    result = runner.invoke(app, ["etl", "run"])
    assert result.exit_code != 0
    assert "Missing option" in result.output or "Error" in result.output


def test_etl_run_nonexistent_file() -> None:
    """Verify etl run with a nonexistent file path fails."""
    result = runner.invoke(app, ["etl", "run", "--path", "does_not_exist_xyz.csv"])
    assert result.exit_code != 0


def test_etl_run_invalid_source(tmp_path: Path) -> None:
    """Verify etl run with an unsupported source adapter fails."""
    dummy = tmp_path / "dummy.csv"
    dummy.write_text("Invoice,StockCode\n123,456\n")

    result = runner.invoke(
        app,
        ["etl", "run", "--source", "unsupported_adapter", "--path", str(dummy)],
    )
    assert result.exit_code == 1
    assert "Unsupported source" in result.output


def test_etl_run_end_to_end_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify full etl run execution on mini_retail fixture with an in-memory database."""
    fixture_path = Path("tests/fixtures/mini_retail.csv")
    assert fixture_path.exists()

    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={"schema_translate_map": {"core": None, "ops": None, "app": None}},
    )
    tables = [
        cast(sa.Table, IngestionBatch.__table__),
        cast(sa.Table, DataQualityResult.__table__),
        cast(sa.Table, DimStore.__table__),
        cast(sa.Table, DimProduct.__table__),
        cast(sa.Table, DimCustomer.__table__),
        cast(sa.Table, DimDate.__table__),
        cast(sa.Table, FactSalesLine.__table__),
        cast(sa.Table, FactDailySales.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)

    @contextmanager
    def mock_session_scope() -> Generator[Session, None, None]:
        with session_factory() as session:
            yield session
            session.commit()

    monkeypatch.setattr("retailpulse.cli.session_scope", mock_session_scope)

    # 1. Execute CLI with --json
    result = runner.invoke(
        app,
        [
            "etl",
            "run",
            "--source",
            "online_retail_ii",
            "--path",
            str(fixture_path),
            "--json",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(result.stdout)

    assert data["status"] == "succeeded"
    assert data["rows_in"] > 0
    assert data["rows_loaded"] > 0
    assert data["daily_records_written"] > 0
    assert data["products_upserted"] > 0
    assert data["data_as_of"] is not None

    # 2. Execute second time without JSON to verify console output and idempotency
    result_console = runner.invoke(
        app,
        [
            "etl",
            "run",
            "--source",
            "online_retail_ii",
            "--path",
            str(fixture_path),
        ],
    )
    assert result_console.exit_code == 0
    assert "ETL completed successfully!" in result_console.output
    assert "Sales lines skipped:" in result_console.output
