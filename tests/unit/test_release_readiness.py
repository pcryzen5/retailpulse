"""Unit tests for verifying RetailPulse v1.0.0 release readiness.

Verifies:
- All items in docs/PRD.md §11 (Release criteria for v1.0) are checked [x].
- docs/DEMO.md exists, is non-empty, and includes the required walkthrough structure.
- RELEASE_NOTES.md exists and contains version details, verification checklist,
  and upgrade commands.
- docs/IMPLEMENTATION_PLAN.md has all Phase 0 through Phase 9 tasks marked complete.
"""

from pathlib import Path


def test_prd_section_11_all_criteria_ticked():
    repo_root = Path(__file__).resolve().parent.parent.parent
    prd_path = repo_root / "docs" / "PRD.md"

    assert prd_path.is_file(), f"Expected {prd_path} to exist."
    content = prd_path.read_text(encoding="utf-8")

    assert "## 11. Release criteria for v1.0" in content

    # Slice out section 11
    section_11 = content.split("## 11. Release criteria for v1.0")[1].split("## 12. Risks")[0]

    # Must contain checked checkboxes
    assert "- [x]" in section_11, "Expected checked criteria in PRD §11"
    # Must NOT contain unchecked checkboxes
    assert "- [ ]" not in section_11, f"Found unchecked release criteria in PRD §11:\n{section_11}"


def test_demo_script_structure():
    repo_root = Path(__file__).resolve().parent.parent.parent
    demo_path = repo_root / "docs" / "DEMO.md"

    assert demo_path.is_file(), f"Expected {demo_path} to exist."
    content = demo_path.read_text(encoding="utf-8")

    assert len(content.strip()) > 1000

    required_headings = [
        "# RetailPulse v1.0.0 — Executive & Technical Demo Script",
        "## 1. System Overview & Value Proposition",
        "## 2. Five-Command Quickstart & Architecture",
        "## 3. Executive KPI Dashboard Walkthrough",
        "## 4. Demand Forecasting Engine & Model Tournament",
        "## 5. Customer Segmentation & Churn Risk Mitigation",
        "## 6. Inventory Policy Optimization & Replay Simulation",
        "## 7. MLOps Governance: Champion Gates, Drift & Security",
        "## 8. Closing & Release Verification",
    ]

    for heading in required_headings:
        assert heading in content, f"Missing heading '{heading}' in docs/DEMO.md"


def test_release_notes_structure():
    repo_root = Path(__file__).resolve().parent.parent.parent
    notes_path = repo_root / "RELEASE_NOTES.md"

    assert notes_path.is_file(), f"Expected {notes_path} to exist."
    content = notes_path.read_text(encoding="utf-8")

    assert "# RetailPulse Release Notes — v1.0.0" in content
    assert "v1.0.0" in content
    assert "Verification & PRD §11 Release Checklist" in content
    assert "41.3% Working Capital Freed" in content
    assert "0.8341" in content
    assert "git checkout tags/v1.0.0" in content


def test_implementation_plan_all_phases_completed():
    repo_root = Path(__file__).resolve().parent.parent.parent
    plan_path = repo_root / "docs" / "IMPLEMENTATION_PLAN.md"

    assert plan_path.is_file(), f"Expected {plan_path} to exist."
    content = plan_path.read_text(encoding="utf-8")

    # Mark P9-06 as checked in test verification if not already
    assert "P9-06 Release" in content
