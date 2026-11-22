"""Reference data and daily sales API endpoints.

Implements API §2.2 and §3:
- GET /api/v1/stores: List stores.
- GET /api/v1/products: Search products with pagination, ABC filter, eligibility flag.
- GET /api/v1/products/{product_id}: Product detail with ABC/XYZ and eligibility explanation.
- GET /api/v1/sales/daily: Daily sales series (product or catalog total,
  store, bounded date range <= 800 days).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Annotated

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from retailpulse.api.dependencies import require_role
from retailpulse.api.pagination import (
    PaginatedResponse,
    PaginationParams,
    apply_sort,
    get_pagination_params,
    paginate_query,
    parse_sort,
)
from retailpulse.db.models.core import DimProduct, DimStore, FactDailySales
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError, NotFoundError

router = APIRouter(tags=["Reference Data & Sales"])


# --- Pydantic Schemas ---


class StoreResponse(BaseModel):
    """Store representation model."""

    model_config = ConfigDict(extra="forbid")

    store_id: int
    store_code: str
    store_name: str
    country: str | None = None


class ProductListItem(BaseModel):
    """Product summary for catalog listing."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    sku: str
    description: str | None = None
    category: str | None = None
    is_product: bool
    unit_cost_est: Decimal | None = None
    abc_class: str | None = None
    xyz_class: str | None = None
    first_sale_date: datetime.date | None = None
    last_sale_date: datetime.date | None = None
    forecast_eligible: bool


class ProductDetailResponse(BaseModel):
    """Detailed product profile with eligibility rationale."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    sku: str
    description: str | None = None
    category: str | None = None
    is_product: bool
    unit_cost_est: Decimal | None = None
    abc_class: str | None = None
    xyz_class: str | None = None
    first_sale_date: datetime.date | None = None
    last_sale_date: datetime.date | None = None
    forecast_eligible: bool
    eligibility_reason: str | None = None


class DailySalesPoint(BaseModel):
    """Aggregated daily sales metrics."""

    model_config = ConfigDict(extra="forbid")

    date: datetime.date
    units_sold: int
    units_returned: int
    net_units: int
    revenue: Decimal
    n_orders: int


# --- Helper Functions ---


def check_product_eligibility(
    product: DimProduct,
    session: Session | None = None,
) -> tuple[bool, str]:
    """Assess whether a product is eligible for time-series forecasting (TRD §5.1).

    Criteria:
    1. Must be a physical merchandise product (is_product == True).
    2. Must have recorded first_sale_date and last_sale_date.
    3. Span between first and last sale date must be >= 90 days.
    4. Must have >= 20 non-zero sales days in trailing year (checked if session provided).
    """
    if not product.is_product:
        return False, "Not eligible: Non-merchandise product (postage, fee, or manual adjustment)"

    if product.first_sale_date is None or product.last_sale_date is None:
        return False, "Not eligible: No sales transaction history recorded"

    span_days = (product.last_sale_date - product.first_sale_date).days
    if span_days < 90:
        return (
            False,
            f"Not eligible: Sales history span of {span_days} days is below the required 90 days",
        )

    if session is not None and product.last_sale_date is not None:
        window_start = product.last_sale_date - datetime.timedelta(days=365)
        sales_days_count = session.scalar(
            sa.select(sa.func.count(FactDailySales.date_key)).where(
                FactDailySales.product_id == product.product_id,
                FactDailySales.date_key >= window_start,
                FactDailySales.date_key <= product.last_sale_date,
                FactDailySales.units_sold > 0,
            )
        )
        if (sales_days_count or 0) < 20:
            return (
                False,
                f"Not eligible: Only {sales_days_count or 0} active sales days "
                "in trailing year (minimum 20 required)",
            )

    return True, "Eligible for forecasting"


# --- Endpoints ---


@router.get(
    "/stores",
    summary="List Stores",
    response_model=list[StoreResponse],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def list_stores(
    session: Annotated[Session, Depends(get_db)],
) -> list[StoreResponse]:
    """List all available store locations."""
    stores = session.execute(sa.select(DimStore).order_by(DimStore.store_id.asc())).scalars().all()
    return [
        StoreResponse(
            store_id=s.store_id,
            store_code=s.store_code,
            store_name=s.store_name,
            country=s.country,
        )
        for s in stores
    ]


@router.get(
    "/products",
    summary="Search Products",
    response_model=PaginatedResponse[ProductListItem],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def list_products(  # noqa: PLR0917
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    q: Annotated[str | None, Query(description="Search term in SKU or description")] = None,
    abc_class: Annotated[str | None, Query(description="Filter by ABC class (A, B, C)")] = None,
    is_product: Annotated[bool | None, Query(description="Filter merchandise items")] = None,
    forecast_eligible: Annotated[
        bool | None, Query(description="Filter forecast-eligible SKUs")
    ] = None,
    sort: Annotated[
        str | None, Query(description="Sort clause, e.g. 'sku:asc' or 'product_id:desc'")
    ] = None,
) -> PaginatedResponse[ProductListItem]:
    """Search and paginate products with filtering."""
    query = sa.select(DimProduct)

    if q:
        term = f"%{q.strip().lower()}%"
        query = query.where(
            sa.or_(
                sa.func.lower(DimProduct.sku).like(term),
                sa.func.lower(DimProduct.description).like(term),
            )
        )

    if abc_class:
        clean_abc = abc_class.strip().upper()
        if clean_abc not in ("A", "B", "C"):
            raise DomainValidationError("abc_class filter must be one of 'A', 'B', or 'C'")
        query = query.where(DimProduct.abc_class == clean_abc)

    if is_product is not None:
        query = query.where(DimProduct.is_product.is_(is_product))

    allowed_sort_fields = {
        "product_id": DimProduct.product_id,
        "sku": DimProduct.sku,
        "description": DimProduct.description,
        "abc_class": DimProduct.abc_class,
        "first_sale_date": DimProduct.first_sale_date,
        "last_sale_date": DimProduct.last_sale_date,
    }

    sort_clauses = parse_sort(sort, allowed_fields=allowed_sort_fields.keys(), default="sku:asc")
    query = apply_sort(query, sort_clauses, allowed_sort_fields)

    products, total = paginate_query(
        query=query,
        session=session,
        page=pagination.page,
        page_size=pagination.page_size,
    )

    items: list[ProductListItem] = []
    for p in products:
        eligible, _ = check_product_eligibility(p)
        if forecast_eligible is not None and eligible != forecast_eligible:
            continue
        items.append(
            ProductListItem(
                product_id=p.product_id,
                sku=p.sku,
                description=p.description,
                category=p.category,
                is_product=p.is_product,
                unit_cost_est=p.unit_cost_est,
                abc_class=p.abc_class,
                xyz_class=p.xyz_class,
                first_sale_date=p.first_sale_date,
                last_sale_date=p.last_sale_date,
                forecast_eligible=eligible,
            )
        )

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.get(
    "/products/{product_id}",
    summary="Get Product Detail",
    response_model=ProductDetailResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_product(
    product_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> ProductDetailResponse:
    """Retrieve complete product profile including ABC/XYZ and forecasting eligibility."""
    product = session.execute(
        sa.select(DimProduct).where(DimProduct.product_id == product_id)
    ).scalar_one_or_none()

    if product is None:
        raise NotFoundError(f"Product {product_id} not found")

    eligible, reason = check_product_eligibility(product, session=session)

    return ProductDetailResponse(
        product_id=product.product_id,
        sku=product.sku,
        description=product.description,
        category=product.category,
        is_product=product.is_product,
        unit_cost_est=product.unit_cost_est,
        abc_class=product.abc_class,
        xyz_class=product.xyz_class,
        first_sale_date=product.first_sale_date,
        last_sale_date=product.last_sale_date,
        forecast_eligible=eligible,
        eligibility_reason=reason,
    )


@router.get(
    "/sales/daily",
    summary="Daily Sales Series",
    response_model=list[DailySalesPoint],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_daily_sales(
    session: Annotated[Session, Depends(get_db)],
    from_date: Annotated[
        datetime.date,
        Query(
            alias="from",
            description="Start date (YYYY-MM-DD)",
        ),
    ],
    to_date: Annotated[
        datetime.date,
        Query(
            alias="to",
            description="End date (YYYY-MM-DD)",
        ),
    ],
    product_id: Annotated[
        int | None,
        Query(description="Product ID (optional; omit for catalog total)"),
    ] = None,
    store_id: Annotated[int | None, Query(description="Store ID")] = None,
) -> list[DailySalesPoint]:
    """Retrieve daily sales time series for a product or total catalog (bounded to 800 days)."""
    if to_date < from_date:
        raise DomainValidationError(
            f"Invalid date range: 'to' ({to_date}) must be >= 'from' ({from_date})"
        )

    span = (to_date - from_date).days
    if span > 800:
        raise DomainValidationError(
            f"Date range span ({span} days) exceeds maximum allowed 800 days"
        )

    # If product_id is specified, return direct daily points
    if product_id is not None:
        query = (
            sa.select(FactDailySales)
            .where(
                FactDailySales.product_id == product_id,
                FactDailySales.date_key >= from_date,
                FactDailySales.date_key <= to_date,
            )
            .order_by(FactDailySales.date_key.asc())
        )
        if store_id is not None:
            query = query.where(FactDailySales.store_id == store_id)

        rows = session.execute(query).scalars().all()
        return [
            DailySalesPoint(
                date=r.date_key,
                units_sold=r.units_sold,
                units_returned=r.units_returned,
                net_units=r.net_units,
                revenue=Decimal(str(r.revenue)),
                n_orders=r.n_orders,
            )
            for r in rows
        ]

    # Aggregate across catalog
    agg_query = (
        sa.select(
            FactDailySales.date_key,
            sa.func.sum(FactDailySales.units_sold).label("units_sold"),
            sa.func.sum(FactDailySales.units_returned).label("units_returned"),
            sa.func.sum(FactDailySales.net_units).label("net_units"),
            sa.func.sum(FactDailySales.revenue).label("revenue"),
            sa.func.sum(FactDailySales.n_orders).label("n_orders"),
        )
        .where(
            FactDailySales.date_key >= from_date,
            FactDailySales.date_key <= to_date,
        )
        .group_by(FactDailySales.date_key)
        .order_by(FactDailySales.date_key.asc())
    )
    if store_id is not None:
        agg_query = agg_query.where(FactDailySales.store_id == store_id)

    agg_rows = session.execute(agg_query).all()
    return [
        DailySalesPoint(
            date=row.date_key,
            units_sold=int(row.units_sold or 0),
            units_returned=int(row.units_returned or 0),
            net_units=int(row.net_units or 0),
            revenue=Decimal(str(row.revenue or 0)),
            n_orders=int(row.n_orders or 0),
        )
        for row in agg_rows
    ]
