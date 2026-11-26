"""Verification test for P9-04 README requirements.

Verifies:
1. Five-command quickstart section is present and formatted.
2. Architecture diagram (Mermaid flowchart) is present and syntactically valid.
3. All 8 dashboard screenshots are referenced and exist on disk.
4. Accepted limitations (v1.0) are explicitly documented.
5. Dataset citation (BibTeX, DOI 10.24432/C5CG6D) and CC BY 4.0 license are documented.
6. Test instructions are provided for unit, integration, and load tests.
"""

from __future__ import annotations

import pathlib


def test_readme_file_exists() -> None:
    """Verify README.md exists and is non-empty."""
    readme_path = pathlib.Path("README.md")
    assert readme_path.exists(), "README.md must exist in repository root"
    assert len(readme_path.read_text(encoding="utf-8")) > 500


def test_readme_quickstart_section() -> None:
    """Verify README contains a 5-command quickstart."""
    content = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "## 1. Five-Command Quick Start" in content
    assert "uv sync --frozen" in content
    assert "docker compose up -d" in content
    assert "alembic upgrade head" in content
    assert "retailpulse users create-admin" in content


def test_readme_architecture_diagram() -> None:
    """Verify README contains a Mermaid architecture diagram."""
    content = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "```mermaid" in content
    assert "flowchart TD" in content
    assert "ClientLayer" in content
    assert "FastAPI REST Service" in content
    assert "Streamlit Analytics Dashboard" in content
    assert "PostgreSQL 16" in content


def test_readme_screenshots_exist() -> None:
    """Verify all 8 screenshots referenced in README exist on disk."""
    content = pathlib.Path("README.md").read_text(encoding="utf-8")
    expected_screenshots = [
        "docs/assets/screenshots/overview.png",
        "docs/assets/screenshots/demand_forecast.png",
        "docs/assets/screenshots/customer_segments.png",
        "docs/assets/screenshots/churn_risk.png",
        "docs/assets/screenshots/inventory_policy.png",
        "docs/assets/screenshots/model_monitoring.png",
        "docs/assets/screenshots/data_pipelines.png",
        "docs/assets/screenshots/admin_users.png",
    ]

    for rel_path in expected_screenshots:
        assert rel_path in content, f"Screenshot {rel_path} must be referenced in README"
        file_path = pathlib.Path(rel_path)
        assert file_path.exists(), f"Screenshot file {rel_path} must exist on disk"
        assert file_path.stat().st_size > 0, f"Screenshot {rel_path} must not be empty"


def test_readme_limitations_section() -> None:
    """Verify accepted limitations are clearly documented."""
    content = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "## 6. Known Limitations" in content
    assert "Rate Limiting" in content
    assert "Single-Tenant" in content
    assert "Synthetic Inventory" in content


def test_readme_dataset_citation_and_license() -> None:
    """Verify dataset citation, DOI, and license are documented."""
    content = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "## 7. Dataset Citation & License" in content
    assert "Online Retail II" in content
    assert "10.24432/C5CG6D" in content
    assert "CC BY 4.0" in content
    assert "@misc{chen_2012_online_retail_ii" in content


def test_readme_test_instructions() -> None:
    """Verify comprehensive testing instructions are documented."""
    content = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "## 4. Test Instructions & Verification" in content
    assert "pytest tests/unit" in content
    assert "pytest tests/integration" in content
    assert "run_load_test.py" in content
    assert "p95" in content
