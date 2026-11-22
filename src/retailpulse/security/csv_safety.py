"""CSV export safety and formula injection mitigation.

Implements SEC-INP-08 and API §4.6:
- Neutralizes formula injection characters (=, +, -, @, \\t, \\r) in text cells.
- Preserves numeric types (int, float, Decimal) without alteration.
- Builds safe CSV rows and sets Content-Disposition: attachment headers.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from decimal import Decimal
from typing import Any

from starlette.responses import StreamingResponse

FORMULA_TRIGGERS: tuple[str, ...] = ("=", "+", "-", "@", "\t", "\r")


def sanitize_csv_cell(value: Any) -> Any:
    """Sanitize a single CSV cell value against spreadsheet formula injection.

    Per SEC-INP-08:
    - Text cells starting with '=', '+', '-', '@', tab, or carriage return
      are prefixed with a single quote (').
    - True numeric types (int, float, Decimal) are not altered.
    - None and booleans are preserved.

    Args:
        value: Cell value of arbitrary type.

    Returns:
        Safe cell value for CSV output.
    """
    if value is None:
        return ""

    # Numeric types and booleans are safe and must not be altered
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return value

    # String / text representation
    val_str = str(value)
    if val_str and val_str[0] in FORMULA_TRIGGERS:
        return f"'{val_str}"

    return val_str


def sanitize_csv_row(row: Sequence[Any]) -> list[Any]:
    """Sanitize all values in a sequence of cells.

    Args:
        row: Sequence of row values.

    Returns:
        List of sanitized values.
    """
    return [sanitize_csv_cell(cell) for cell in row]


def format_csv(
    rows: Iterable[Sequence[Any]],
    headers: Sequence[str] | None = None,
) -> str:
    """Render rows to a safe CSV string.

    Args:
        rows: Iterable of cell sequences.
        headers: Optional sequence of column headers.

    Returns:
        CSV text formatted with formula sanitization.
    """
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\r\n")

    if headers is not None:
        # Headers are text labels; sanitize in case headers themselves contain trigger characters
        writer.writerow(sanitize_csv_row(headers))

    for row in rows:
        writer.writerow(sanitize_csv_row(row))

    return output.getvalue()


def create_csv_streaming_response(
    rows: Iterable[Sequence[Any]],
    filename: str,
    headers: Sequence[str] | None = None,
) -> StreamingResponse:
    """Create a FastAPI/Starlette StreamingResponse for CSV export.

    Sets Content-Disposition: attachment; filename="{filename}"
    and Content-Type: text/csv; charset=utf-8.

    Args:
        rows: Iterable of data rows.
        filename: Destination download filename (e.g. 'customers_export.csv').
        headers: Optional column header labels.

    Returns:
        StreamingResponse configured with safe CSV stream.
    """
    csv_content = format_csv(rows, headers=headers)

    safe_filename = filename.replace('"', "").replace("\r", "").replace("\n", "")

    headers_dict = {
        "Content-Disposition": f'attachment; filename="{safe_filename}"',
        "Content-Type": "text/csv; charset=utf-8",
    }

    return StreamingResponse(
        iter([csv_content.encode("utf-8")]),
        media_type="text/csv",
        headers=headers_dict,
    )
