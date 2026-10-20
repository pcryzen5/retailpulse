"""Unit tests for the RetailPulse CLI."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from retailpulse import __version__
from retailpulse.cli import app

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
