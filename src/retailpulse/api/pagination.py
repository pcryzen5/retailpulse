"""Pagination, sorting whitelist, and list envelope helpers.

Implements API §1.1 (Pagination and Sorting) and SEC-INP-01:
- Pagination parameters: page >= 1, page_size between 1 and 200 (default 50).
- Response envelope: {"items": [...], "page": 1, "page_size": 50, "total": 1234}.
- Whitelisted sorting parser and query builder (format: 'field:asc' or 'field:desc').
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import sqlalchemy as sa
from fastapi import Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from retailpulse.errors import DomainValidationError

T = TypeVar("T")

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200


class PaginationParams(BaseModel):
    """Pagination query parameters model with validation bounds (SEC-INP-01)."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(
        default=DEFAULT_PAGE,
        ge=1,
        description="Page number (1-based index).",
    )
    page_size: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Number of items per page (1 to {MAX_PAGE_SIZE}).",
    )

    @property
    def offset(self) -> int:
        """Calculate SQL query offset."""
        return (self.page - 1) * self.page_size

    @property
    def limit(self) -> int:
        """Calculate SQL query limit."""
        return self.page_size


def get_pagination_params(
    page: int = Query(
        default=DEFAULT_PAGE,
        ge=1,
        description="Page number (1-based).",
    ),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE}).",
    ),
) -> PaginationParams:
    """FastAPI query dependency providing validated pagination parameters."""
    return PaginationParams(page=page, page_size=page_size)


class PaginatedResponse(BaseModel, Generic[T]):
    """Standard API list envelope (API §1.1)."""

    model_config = ConfigDict(extra="forbid")

    items: list[T]
    page: int
    page_size: int
    total: int


@dataclass(frozen=True)
class SortClause:
    """Parsed sort instruction."""

    field: str
    direction: Literal["asc", "desc"]


def parse_sort(
    sort_param: str | None,
    allowed_fields: Collection[str],
    default: str | None = None,
) -> list[SortClause]:
    """Parse and validate sort parameter against an allowed fields whitelist.

    Accepts expressions such as:
    - 'churn_probability:desc'
    - 'monetary:asc,recency_days:desc'
    - 'field' (defaults to 'asc' if direction omitted)

    Args:
        sort_param: Raw query string parameter.
        allowed_fields: Allowlist of valid sortable field names.
        default: Fallback sort expression when sort_param is None or empty.

    Returns:
        List of validated SortClause instances.

    Raises:
        DomainValidationError: If field is unauthorized or direction is malformed.
    """
    expr = sort_param or default
    if not expr:
        return []

    clauses: list[SortClause] = []
    tokens = [t.strip() for t in expr.split(",") if t.strip()]

    for token in tokens:
        if ":" in token:
            parts = token.split(":", 1)
            field = parts[0].strip()
            direction = parts[1].strip().lower()
        else:
            field = token.strip()
            direction = "asc"

        if field not in allowed_fields:
            sorted_allowed = sorted(allowed_fields)
            raise DomainValidationError(
                f"Invalid sort field '{field}'. Allowed fields: {sorted_allowed}"
            )

        if direction not in ("asc", "desc"):
            raise DomainValidationError(
                f"Invalid sort direction '{direction}' for field '{field}'. Must be 'asc' or 'desc'"
            )

        clauses.append(SortClause(field=field, direction=direction))  # type: ignore[arg-type]

    return clauses


def apply_sort(
    query: sa.Select[Any],
    sort_clauses: Sequence[SortClause],
    column_map: Mapping[str, Any],
) -> sa.Select[Any]:
    """Apply validated sort clauses to a SQLAlchemy select query.

    Args:
        query: Base SQLAlchemy select query.
        sort_clauses: Sequence of validated sort clauses.
        column_map: Mapping from field name to SQLAlchemy column or expression.

    Returns:
        Query with order_by clauses applied.
    """
    for clause in sort_clauses:
        column = column_map.get(clause.field)
        if column is not None:
            if clause.direction == "desc":
                query = query.order_by(sa.desc(column))
            else:
                query = query.order_by(sa.asc(column))
    return query


def paginate_sequence(
    items: Sequence[T],
    page: int,
    page_size: int,
) -> PaginatedResponse[T]:
    """Paginate an in-memory sequence into a PaginatedResponse envelope.

    Args:
        items: Full sequence of items.
        page: Page number (1-based).
        page_size: Page size.

    Returns:
        PaginatedResponse envelope containing sliced items.
    """
    total = len(items)
    offset = (page - 1) * page_size
    sliced = list(items[offset : offset + page_size])
    return PaginatedResponse(
        items=sliced,
        page=page,
        page_size=page_size,
        total=total,
    )


def paginate_query(
    query: sa.Select[Any],
    session: Session,
    page: int,
    page_size: int,
) -> tuple[list[Any], int]:
    """Execute count and offset/limit pagination on a SQLAlchemy select query.

    Args:
        query: Base SQLAlchemy query without limit/offset.
        session: Active database session.
        page: Page number (1-based).
        page_size: Number of items per page.

    Returns:
        Tuple of (items, total_count).
    """
    # Total count query
    count_subquery = query.order_by(None).subquery()
    count_query = sa.select(sa.func.count()).select_from(count_subquery)
    total = session.scalar(count_query) or 0

    # Paginated data query
    offset = (page - 1) * page_size
    paged_query = query.offset(offset).limit(page_size)
    items = list(session.execute(paged_query).scalars().all())

    return items, total

