"""Unit and CLI tests for retailpulse users commands.

Implements P5-02 CLI verification:
- retailpulse users create with password policy and Argon2id hashing.
- retailpulse users list (text table and JSON output).
- retailpulse users deactivate with last active admin protection.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.db.models.app import AppUser
from retailpulse.errors import ConflictError
from retailpulse.security.password import hash_password

runner = CliRunner()


def test_cli_users_help() -> None:
    """Verify retailpulse users --help displays all commands."""
    result = runner.invoke(app, ["users", "--help"])
    assert result.exit_code == 0
    assert "create" in result.output
    assert "list" in result.output
    assert "deactivate" in result.output


@patch("retailpulse.cli.session_scope")
def test_cli_users_create_success(mock_scope: MagicMock) -> None:
    """Verify retailpulse users create adds new user to session."""
    mock_session = MagicMock()
    mock_scope.return_value.__enter__.return_value = mock_session
    mock_session.execute.return_value.scalar_one_or_none.return_value = None

    result = runner.invoke(
        app,
        [
            "users",
            "create",
            "--email",
            "admin@example.com",
            "--role",
            "admin",
            "--full-name",
            "Admin User",
            "--password",
            "ValidPassword123!",
        ],
    )
    assert result.exit_code == 0
    assert "Successfully created user admin@example.com" in result.output
    assert mock_session.add.called
    assert mock_session.commit.called


@patch("retailpulse.cli.session_scope")
def test_cli_users_create_invalid_role(mock_scope: MagicMock) -> None:
    """Verify invalid role is rejected."""
    result = runner.invoke(
        app,
        [
            "users",
            "create",
            "--email",
            "analyst@example.com",
            "--role",
            "superuser",
            "--password",
            "ValidPassword123!",
        ],
    )
    assert result.exit_code == 1
    assert "Invalid role" in result.output


@patch("retailpulse.cli.session_scope")
def test_cli_users_create_weak_password(mock_scope: MagicMock) -> None:
    """Verify passwords failing policy are rejected."""
    result = runner.invoke(
        app,
        [
            "users",
            "create",
            "--email",
            "analyst@example.com",
            "--role",
            "analyst",
            "--password",
            "short",
        ],
    )
    assert result.exit_code == 1
    assert "Password Error" in result.output


@patch("retailpulse.cli.session_scope")
def test_cli_users_list_json(mock_scope: MagicMock) -> None:
    """Verify retailpulse users list --json returns valid JSON records."""
    mock_session = MagicMock()
    mock_scope.return_value.__enter__.return_value = mock_session

    u = AppUser(
        email="viewer@example.com",
        role="viewer",
        full_name="Viewer One",
        password_hash=hash_password("ValidPassword123!"),
        is_active=True,
    )
    mock_session.execute.return_value.scalars.return_value.all.return_value = [u]

    result = runner.invoke(app, ["users", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert len(data) == 1
    assert data[0]["email"] == "viewer@example.com"
    assert data[0]["role"] == "viewer"


@patch("retailpulse.cli.session_scope")
@patch("retailpulse.cli.ensure_not_last_admin")
def test_cli_users_deactivate_last_admin_fails(
    mock_ensure_admin: MagicMock,
    mock_scope: MagicMock,
) -> None:
    """Verify deactivating last active admin is prevented."""
    mock_session = MagicMock()
    mock_scope.return_value.__enter__.return_value = mock_session

    u = AppUser(
        email="admin@example.com",
        role="admin",
        password_hash="...",
        is_active=True,
    )
    mock_session.execute.return_value.scalar_one_or_none.return_value = u
    mock_ensure_admin.side_effect = ConflictError(
        "Cannot demote or deactivate the last active administrator"
    )

    result = runner.invoke(app, ["users", "deactivate", "admin@example.com"])
    assert result.exit_code == 1
    assert "Cannot demote or deactivate the last active administrator" in result.output
