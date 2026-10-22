"""Integration and CLI tests for retailpulse db check and database connectivity verification."""

from __future__ import annotations

import json

import pytest
import sqlalchemy as sa
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.config import get_settings
from retailpulse.db.session import check_database
from retailpulse.errors import DatabaseError

runner = CliRunner()


def test_cli_db_help() -> None:
    """Verify retailpulse db --help displays check command."""
    result = runner.invoke(app, ["db", "--help"])
    assert result.exit_code == 0
    assert "check" in result.output


def test_check_database_unreachable() -> None:
    """Verify check_database raises DatabaseError on invalid connection string."""
    bad_engine = sa.create_engine(
        "postgresql+psycopg://baduser:badpass@localhost:9999/baddb",
        connect_args={"connect_timeout": 1},
    )
    with pytest.raises(DatabaseError):
        check_database(engine=bad_engine)


def test_cli_db_check_unreachable_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify retailpulse db check --json outputs structured failure when DB is unreachable."""
    monkeypatch.setenv(
        "RP_DATABASE_URL",
        "postgresql+psycopg://baduser:badpass@localhost:9999/baddb",
    )
    result = runner.invoke(app, ["db", "check", "--json"])
    assert result.exit_code == 1
    json_lines = [
        line.strip() for line in result.output.splitlines() if line.strip().startswith("{")
    ]
    assert len(json_lines) > 0
    data = json.loads(json_lines[-1])
    assert data["connected"] is False
    assert "error" in data


@pytest.mark.integration
def test_live_db_check_connected() -> None:
    """Verify check_database reports status against a live PostgreSQL server when available."""
    settings = get_settings()
    try:
        engine = sa.create_engine(
            settings.database_url,
            connect_args={"connect_timeout": 1},
        )
        with engine.connect():
            pass
    except Exception as exc:
        pytest.skip(f"Live PostgreSQL instance not reachable ({exc}); skipping live db check.")

    status = check_database(engine=engine)
    assert status["connected"] is True
    assert "head_revision" in status
    assert status["head_revision"] == "0004"
