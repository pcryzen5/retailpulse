"""Unit tests for test fixture generation and deterministic repeatability."""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.make_fixture import TARGET_PATH, generate_fixture


def test_fixture_generation_deterministic(tmp_path: Path) -> None:
    """Verify that fixture generation produces identical bytes across runs."""
    path1 = tmp_path / "fixture1.csv"
    path2 = tmp_path / "fixture2.csv"

    generate_fixture(output_path=path1)
    generate_fixture(output_path=path2)

    bytes1 = path1.read_bytes()
    bytes2 = path2.read_bytes()

    assert bytes1 == bytes2
    assert hashlib.sha256(bytes1).hexdigest() == hashlib.sha256(bytes2).hexdigest()
    assert path1.stat().st_size > 0


def test_fixture_contents() -> None:
    """Verify that the generated fixture contains expected edge case patterns."""
    assert TARGET_PATH.exists()
    content = TARGET_PATH.read_text(encoding="utf-8")

    # Cancellations
    assert "C541433" in content
    # Outlier quantity
    assert "74215" in content
    # Non-product SKUs
    assert "BANK CHARGES" in content
    assert "POST" in content
    # Real product with DCGS prefix
    assert "DCGS0003" in content
    # Invalid cancellation with positive qty
    assert "C539999" in content
    # Case variants
    assert "85123a" in content
    assert "85123A" in content
    # Partial last day
    assert "2011-12-09 12:50:00" in content
    # Bad unparseable row
    assert "INVALID-DATE-TIME" in content
