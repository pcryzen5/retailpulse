"""Unit tests for API pagination, sorting whitelist, and response envelope.

Implements API §1.1 and SEC-INP-01 verification:
- Pagination parameters validation bounds (page >= 1, 1 <= page_size <= 200).
- Response envelope shape (items, page, page_size, total).
- Sequence pagination.
- Sort string parsing, allowed-field whitelisting, and direction validation.
- SQLAlchemy sorting and database query pagination.
"""

from __future__ import annotations

import pytest
import sqlalchemy as sa
from pydantic import ValidationError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from retailpulse.api.pagination import (
    DEFAULT_PAGE,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    PaginatedResponse,
    PaginationParams,
    SortClause,
    apply_sort,
    paginate_query,
    paginate_sequence,
    parse_sort,
)
from retailpulse.errors import DomainValidationError

# --- Database fixtures for query pagination and sort tests ---

test_meta = sa.MetaData()


class BaseItem(DeclarativeBase):
    metadata = test_meta


class ProductItem(BaseItem):
    __tablename__ = "test_pagination_items"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    name: Mapped[str] = mapped_column(sa.String, nullable=False)
    price: Mapped[float] = mapped_column(sa.Float, nullable=False)


@pytest.fixture
def sqlite_session():
    """In-memory SQLite session seeded with sample items."""
    engine = sa.create_engine("sqlite:///:memory:")
    test_meta.create_all(engine)

    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    items = [
        ProductItem(id=1, name="Alpha", price=10.0),
        ProductItem(id=2, name="Beta", price=25.0),
        ProductItem(id=3, name="Gamma", price=5.0),
        ProductItem(id=4, name="Delta", price=50.0),
        ProductItem(id=5, name="Epsilon", price=15.0),
    ]
    session.add_all(items)
    session.commit()

    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# --- 1. PaginationParams Bounds Tests (SEC-INP-01) ---


def test_pagination_defaults() -> None:
    """Verify default pagination parameters."""
    p = PaginationParams()
    assert p.page == DEFAULT_PAGE
    assert p.page_size == DEFAULT_PAGE_SIZE
    assert p.offset == 0
    assert p.limit == 50


def test_pagination_custom_valid() -> None:
    """Verify valid custom pagination bounds."""
    p = PaginationParams(page=3, page_size=25)
    assert p.page == 3
    assert p.page_size == 25
    assert p.offset == 50
    assert p.limit == 25


def test_pagination_invalid_page_bounds() -> None:
    """Verify page < 1 is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        PaginationParams(page=0)

    with pytest.raises(ValidationError):
        PaginationParams(page=-5)


def test_pagination_invalid_page_size_bounds() -> None:
    """Verify page_size < 1 or > MAX_PAGE_SIZE (200) is rejected."""
    with pytest.raises(ValidationError):
        PaginationParams(page_size=0)

    with pytest.raises(ValidationError):
        PaginationParams(page_size=MAX_PAGE_SIZE + 1)


def test_pagination_forbid_extra() -> None:
    """Verify extra query parameters are rejected."""
    with pytest.raises(ValidationError):
        PaginationParams.model_validate({"page": 1, "page_size": 50, "unauthorized_extra": "drop"})


# --- 2. Response Envelope Tests (API §1.1) ---


def test_paginated_response_envelope_serialization() -> None:
    """Verify response envelope adheres to API §1.1 specification."""
    envelope: PaginatedResponse[str] = PaginatedResponse(
        items=["item1", "item2"],
        page=1,
        page_size=50,
        total=2,
    )
    data = envelope.model_dump()
    assert data == {
        "items": ["item1", "item2"],
        "page": 1,
        "page_size": 50,
        "total": 2,
    }


def test_paginate_sequence_in_memory() -> None:
    """Verify in-memory sequence pagination."""
    items = list(range(100))

    # Page 1 (items 0 to 9)
    resp1 = paginate_sequence(items, page=1, page_size=10)
    assert resp1.page == 1
    assert resp1.page_size == 10
    assert resp1.total == 100
    assert resp1.items == list(range(10))

    # Page 10 (items 90 to 99)
    resp10 = paginate_sequence(items, page=10, page_size=10)
    assert resp10.items == list(range(90, 100))

    # Page beyond available items
    resp_empty = paginate_sequence(items, page=11, page_size=10)
    assert resp_empty.items == []
    assert resp_empty.total == 100


# --- 3. Sorting Whitelist Tests (SEC-INP-01) ---


def test_parse_sort_valid_single_clause() -> None:
    """Verify parsing single sort clause with explicit direction."""
    allowed = {"price", "name", "created_at"}
    clauses = parse_sort("price:desc", allowed_fields=allowed)
    assert clauses == [SortClause(field="price", direction="desc")]


def test_parse_sort_default_direction() -> None:
    """Verify sort clause without explicit direction defaults to 'asc'."""
    allowed = {"price", "name"}
    clauses = parse_sort("name", allowed_fields=allowed)
    assert clauses == [SortClause(field="name", direction="asc")]


def test_parse_sort_multiple_clauses() -> None:
    """Verify comma-separated multiple sort clauses."""
    allowed = {"price", "name", "id"}
    clauses = parse_sort("price:desc,name:asc,id:desc", allowed_fields=allowed)
    assert clauses == [
        SortClause(field="price", direction="desc"),
        SortClause(field="name", direction="asc"),
        SortClause(field="id", direction="desc"),
    ]


def test_parse_sort_fallback_to_default() -> None:
    """Verify fallback when sort string is None or empty."""
    allowed = {"id", "created_at"}
    clauses = parse_sort(None, allowed_fields=allowed, default="id:desc")
    assert clauses == [SortClause(field="id", direction="desc")]


def test_parse_sort_unauthorized_field_raises_validation_error() -> None:
    """Verify unauthorized sort fields outside the whitelist are rejected."""
    allowed = {"price", "name"}
    with pytest.raises(DomainValidationError, match="Invalid sort field 'secret_column'"):
        parse_sort("secret_column:asc", allowed_fields=allowed)


def test_parse_sort_invalid_direction_raises_validation_error() -> None:
    """Verify invalid sort direction is rejected."""
    allowed = {"price", "name"}
    with pytest.raises(DomainValidationError, match="Invalid sort direction 'upwards'"):
        parse_sort("price:upwards", allowed_fields=allowed)


# --- 4. Database Query Pagination & Sorting Tests ---


def test_paginate_query_and_apply_sort(sqlite_session) -> None:
    """Verify database query pagination and sorting integration."""
    base_query = sa.select(ProductItem)
    column_map = {"price": ProductItem.price, "name": ProductItem.name}

    # Sort by price descending
    sort_clauses = parse_sort("price:desc", allowed_fields=column_map.keys())
    sorted_query = apply_sort(base_query, sort_clauses, column_map)

    # Paginate page 1, page_size 2
    items, total = paginate_query(sorted_query, sqlite_session, page=1, page_size=2)
    assert total == 5
    assert len(items) == 2
    # Highest prices: Delta (50.0), Beta (25.0)
    assert items[0].name == "Delta"
    assert items[1].name == "Beta"

    # Page 2, page_size 2
    items_p2, total2 = paginate_query(sorted_query, sqlite_session, page=2, page_size=2)
    assert total2 == 5
    assert len(items_p2) == 2
    # Next highest: Epsilon (15.0), Alpha (10.0)
    assert items_p2[0].name == "Epsilon"
    assert items_p2[1].name == "Alpha"
