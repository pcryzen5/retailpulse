"""Verification test for P8-06 Secrets handling and configuration hygiene.

Verifies:
1. .env is ignored in .gitignore (SEC-CFG-01).
2. .env.example contains generation commands, explanations, and placeholder secrets (SEC-CFG-01).
3. Production configuration validation:
   Settings enforces min length and rejects example secrets when RP_ENV=prod (SEC-CFG-02).
4. README documents chmod 0600 file permissions and rotation procedures (SECURITY §8, §12).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from retailpulse.config import Settings
from retailpulse.errors import ConfigError


def test_env_is_git_ignored() -> None:
    """Verify that .env is explicitly excluded in .gitignore (SEC-CFG-01)."""
    gitignore_path = Path(".gitignore")
    assert gitignore_path.exists(), ".gitignore must exist"

    lines = gitignore_path.read_text(encoding="utf-8").splitlines()
    cleaned = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    assert any(line == ".env" or line.startswith(".env") for line in cleaned)


def test_env_example_has_generation_commands_and_placeholders() -> None:
    """Verify that .env.example documents secret generation and safe defaults (SEC-CFG-01)."""
    example_path = Path(".env.example")
    assert example_path.exists(), ".env.example must exist"

    content = example_path.read_text(encoding="utf-8")
    assert "import secrets" in content
    assert "secrets.token_urlsafe" in content
    assert "chmod 0600 .env" in content
    assert "RP_JWT_SECRET=" in content
    assert "RP_PSEUDONYM_SALT=" in content


def test_production_settings_rejects_weak_or_example_secrets() -> None:
    """Verify that RP_ENV=prod rejects weak, short, or placeholder example secrets (SEC-CFG-02)."""
    # 1. Reject placeholder example secret in prod
    with pytest.raises((ValidationError, ConfigError)):
        Settings(
            env="prod",
            jwt_secret="insecure-dev-jwt-secret-min-32-bytes-long!",
            pseudonym_salt="another-random-string-at-least-32-characters!",
        )

    # 2. Reject short secret (< 32 chars) in prod
    with pytest.raises((ValidationError, ConfigError)):
        Settings(
            env="prod",
            jwt_secret="too-short-secret",
            pseudonym_salt="another-random-string-at-least-32-characters!",
        )

    # 3. Accept strong, random secrets in prod
    prod_settings = Settings(
        env="prod",
        jwt_secret="k9Yv3P_xL1mQ8wZ0jR5tU7eA2sD4fG6hJ8kL0zX2cV4bN6mQ8wE0rT2yU4iO6pA8",
        pseudonym_salt="m8K0zX2cV4bN6mQ8wE0rT2yU4iO6pA8k9Yv3P_xL1mQ8wZ0jR5tU7eA2sD4fG6hJ",
    )
    assert prod_settings.env == "prod"
    assert prod_settings.jwt_secret.get_secret_value().startswith("k9Yv3P")


def test_readme_documents_rotation_and_file_mode() -> None:
    """Verify README.md contains rotation instructions and 0600 file mode."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md must exist"

    content = readme_path.read_text(encoding="utf-8")
    assert "chmod 0600 .env" in content
    assert "Secrets Handling & Rotation" in content
    assert "JWT Secret" in content
    assert "Pseudonymization Salt" in content
    assert "Database Passwords" in content
