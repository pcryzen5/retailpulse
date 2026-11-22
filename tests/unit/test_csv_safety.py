"""Unit tests for CSV export safety and formula injection mitigation.

Implements SEC-INP-08 and API §4.6 verification:
- Neutralizing formula injection characters (=, +, -, @, \\t, \\r).
- Preserving pure numeric values (int, float, Decimal) without alteration.
- Preserving booleans and None values.
- Row and tabular format_csv generation.
- create_csv_streaming_response headers and content.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from retailpulse.security.csv_safety import (
    create_csv_streaming_response,
    format_csv,
    sanitize_csv_cell,
    sanitize_csv_row,
)


@pytest.mark.parametrize(
    ("input_val", "expected_val"),
    [
        # Standard text without injection triggers
        ("Product Alpha", "Product Alpha"),
        ("SKU_98765", "SKU_98765"),
        ("electronics", "electronics"),
        # Empty string
        ("", ""),
        # None
        (None, ""),
        # Formula injection characters at start of string
        ("=1+1", "'=1+1"),
        ("=SUM(A1:A10)", "'=SUM(A1:A10)"),
        ("=cmd|' /C calc'!A0", "'=cmd|' /C calc'!A0"),
        ("+12345", "'+12345"),
        ("+cmd", "'+cmd"),
        ("-10% Discount", "'-10% Discount"),
        ("-2+5", "'-2+5"),
        ("@SUM", "'@SUM"),
        ("@evil.com", "'@evil.com"),
        ("\tTabIndented", "'\tTabIndented"),
        ("\rCarriageReturn", "'\rCarriageReturn"),
        # Internal formula characters (not at start) must NOT be prefixed
        ("test=value", "test=value"),
        ("A+B", "A+B"),
        ("email@example.com", "email@example.com"),
    ],
)
def test_sanitize_csv_cell_text_and_formulas(input_val: str | None, expected_val: str) -> None:
    """Verify formula injection triggers are escaped while regular text is untouched."""
    assert sanitize_csv_cell(input_val) == expected_val


@pytest.mark.parametrize(
    ("numeric_val", "expected_val"),
    [
        (0, 0),
        (42, 42),
        (-100, -100),
        (3.14159, 3.14159),
        (-0.05, -0.05),
        (Decimal("199.99"), Decimal("199.99")),
        (Decimal("-50.00"), Decimal("-50.00")),
        (True, True),
        (False, False),
    ],
)
def test_sanitize_csv_cell_preserves_numbers_and_bools(
    numeric_val: int | float | Decimal | bool,
    expected_val: int | float | Decimal | bool,
) -> None:
    """Verify numeric types and booleans are preserved without quote prefixing."""
    result = sanitize_csv_cell(numeric_val)
    assert result == expected_val
    assert type(result) is type(expected_val)


def test_sanitize_csv_row() -> None:
    """Verify all cells in a row are sanitized."""
    raw_row = ["=DDE(1,2)", 42, -9.99, "Normal", "@admin"]
    sanitized = sanitize_csv_row(raw_row)
    assert sanitized == ["'=DDE(1,2)", 42, -9.99, "Normal", "'@admin"]


def test_format_csv_with_headers_and_data() -> None:
    """Verify format_csv produces correct CSV syntax with formula escaping."""
    headers = ["ID", "Name", "Formula", "Price"]
    rows = [
        [1, "Widget A", "=1+1", 19.99],
        [2, "-Negative Label", "+Command", -5.5],
    ]
    csv_text = format_csv(rows, headers=headers)

    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    assert len(lines) == 3
    assert lines[0] == "ID,Name,Formula,Price"
    assert lines[1] == "1,Widget A,'=1+1,19.99"
    assert lines[2] == "2,'-Negative Label,'+Command,-5.5"


@pytest.mark.anyio
async def test_create_csv_streaming_response() -> None:
    """Verify create_csv_streaming_response sets proper headers and content."""
    headers = ["SKU", "Title", "Trigger"]
    rows = [["SKU001", "Item A", "=SUM(B1:B2)"]]

    response = create_csv_streaming_response(
        rows=rows,
        filename="export_customers.csv",
        headers=headers,
    )

    assert response.status_code == 200
    assert response.media_type == "text/csv"
    assert response.headers["Content-Disposition"] == 'attachment; filename="export_customers.csv"'
    assert "text/csv" in response.headers["Content-Type"]

    chunks = [chunk async for chunk in response.body_iterator]
    body = b"".join(chunks).decode("utf-8")
    assert "SKU,Title,Trigger" in body
    assert "SKU001,Item A,'=SUM(B1:B2)" in body
