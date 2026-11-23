"""Inventory control, recommendations, parameters, what-if, and simulation router.

Implements API §2.5, §3.6-§3.8, TRD §5.3:
- GET /api/v1/inventory/summary: Status counts, risk histogram, synthetic params share (Viewer+).
- GET /api/v1/inventory/recommendations: Paginated & filtered recommendations (Viewer+).
- GET /api/v1/inventory/recommendations/export: Formula-safe CSV export (Analyst+ only).
- GET /api/v1/inventory/params/{product_id}: View product policy parameters (Viewer+).
- PUT /api/v1/inventory/params/{product_id}: Upsert policy parameters, source='user' (Analyst+).
- POST /api/v1/inventory/what-if: Recompute policy without database persistence (Analyst+).
- POST /api/v1/inventory/simulate: Synchronous policy simulation with max 200 limit (Analyst+).
- GET /api/v1/inventory/simulations/latest: Latest stored simulation run (Viewer+).
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from retailpulse.api.dependencies import require_role
from retailpulse.api.limiter import limiter
from retailpulse.api.pagination import (
    PaginationParams,
    apply_sort,
    get_pagination_params,
    parse_sort,
)
from retailpulse.db.models.app import AppUser
from retailpulse.db.models.core import DimProduct, InventoryParam, InventorySnapshot
from retailpulse.db.models.ml import (
    ActiveOutput,
    InventoryRecommendation,
    PolicySimulation,
)
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError, NotFoundError
from retailpulse.ml.inventory.policy import evaluate_inventory_policy
from retailpulse.ml.inventory.simulate import SimulationConfig, run_policy_simulation
from retailpulse.security.audit import log_audit
from retailpulse.security.csv_safety import create_csv_streaming_response
from retailpulse.timeutil import utc_now

router = APIRouter(tags=["Inventory"])

TASK_INVENTORY = "inventory"
MAX_SIMULATION_LIMIT = 200

RECOMMENDATION_SORT_WHITELIST: dict[str, Any] = {
    "stockout_risk": InventoryRecommendation.expected_stockout_prob,
    "days_of_cover": InventoryRecommendation.days_of_cover,
    "reorder_urgency": InventoryRecommendation.reorder_urgency,
    "recommended_order_qty": InventoryRecommendation.order_quantity,
    "product_id": InventoryRecommendation.product_id,
    "sku": DimProduct.sku,
}


# --- Schemas ---


class HistogramBin(BaseModel):
    """Single bin in stockout risk probability histogram."""

    model_config = ConfigDict(extra="forbid")

    bin_start: float
    bin_end: float
    count: int


class InventorySummaryResponse(BaseModel):
    """Aggregate inventory summary metrics matching API §2.5."""

    model_config = ConfigDict(extra="forbid")

    as_of_date: datetime.date | None = None
    total_items: int = 0
    status_counts: dict[str, int]
    stockout_risk_histogram: list[HistogramBin]
    params_synthetic_share: float = 0.0


class InventoryRecommendationItem(BaseModel):
    """Inventory recommendation record matching API §3.6."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    sku: str
    description: str | None = None
    abc_class: str | None = None
    store_id: int
    as_of_date: datetime.date
    on_hand: int
    on_order: int
    lead_time_days: float
    service_level: float
    demand_mean_lt: float | None = None
    demand_std_lt: float | None = None
    safety_stock: float
    reorder_point: float
    recommended_order_qty: int
    stockout_risk: float
    days_of_cover: float | None = None
    status: str
    params_source: str


class PaginatedInventoryRecommendationsResponse(BaseModel):
    """Paginated envelope for recommendations matching API §3.6."""

    model_config = ConfigDict(extra="forbid")

    items: list[InventoryRecommendationItem]
    page: int
    page_size: int
    total: int
    pages: int
    params_synthetic_share: float = 0.0


class InventoryParamResponse(BaseModel):
    """Policy parameters for a single product and store."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    store_id: int
    lead_time_days: float
    lead_time_std_days: float
    service_level: float
    unit_cost: float | None = None
    holding_cost_rate: float
    order_cost: float
    moq: int
    source: str
    updated_at: datetime.datetime


class UpdateInventoryParamRequest(BaseModel):
    """Payload for updating inventory policy parameters."""

    model_config = ConfigDict(extra="forbid")

    lead_time_days: float = Field(..., gt=0, description="Supplier lead time in days")
    lead_time_std_days: float = Field(
        default=0.0, ge=0, description="Lead time standard deviation in days"
    )
    service_level: float = Field(
        ..., gt=0.5, lt=1.0, description="Target cycle service level (0.50 to 1.00)"
    )
    unit_cost: float | None = Field(default=None, ge=0, description="Estimated unit purchase cost")
    holding_cost_rate: float = Field(
        default=0.20, ge=0, description="Annual holding cost fraction (e.g. 0.20)"
    )
    order_cost: float = Field(default=25.0, ge=0, description="Fixed order placement cost")
    moq: int = Field(default=1, ge=1, description="Minimum order quantity")


class WhatIfRequest(BaseModel):
    """What-if simulation query for a single product."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    store_id: int = 1
    overrides: dict[str, Any] = Field(default_factory=dict)


class WhatIfResponse(BaseModel):
    """What-if simulation result with non-persistence guarantee."""

    model_config = ConfigDict(extra="forbid")

    product_id: int
    sku: str
    description: str | None = None
    abc_class: str | None = None
    store_id: int
    as_of_date: datetime.date
    on_hand: int
    on_order: int
    lead_time_days: float
    service_level: float
    demand_mean_lt: float | None = None
    demand_std_lt: float | None = None
    safety_stock: float
    reorder_point: float
    recommended_order_qty: int
    stockout_risk: float
    days_of_cover: float | None = None
    status: str
    params_source: str
    persisted: bool = False


class SimulationScope(BaseModel):
    """Scope settings for simulation run."""

    model_config = ConfigDict(extra="forbid")

    abc_class: str | None = None
    limit: int = Field(default=100, ge=1, description="Max SKUs to simulate (<= 200)")


class SimulationPolicy(BaseModel):
    """Policy settings for simulation run."""

    model_config = ConfigDict(extra="forbid")

    service_level: float = Field(default=0.95, gt=0.5, lt=1.0)


class SimulationBaseline(BaseModel):
    """Baseline policy settings for simulation comparison."""

    model_config = ConfigDict(extra="forbid")

    type: str = "moving_average_28"
    cover_days_extra: int = Field(default=14, ge=0)


class SimulationRequest(BaseModel):
    """Request payload for synchronous policy simulation matching API §3.8."""

    model_config = ConfigDict(extra="forbid")

    scope: SimulationScope = Field(default_factory=SimulationScope)
    policy: SimulationPolicy = Field(default_factory=SimulationPolicy)
    baseline: SimulationBaseline = Field(default_factory=SimulationBaseline)


class SimulationWindow(BaseModel):
    """Date window for simulation results."""

    model_config = ConfigDict(extra="forbid")

    start: datetime.date
    end: datetime.date


class SimulationResponse(BaseModel):
    """Simulation run result matching API §3.8."""

    model_config = ConfigDict(extra="forbid")

    simulation_id: uuid.UUID
    window: SimulationWindow
    uses_synthetic_params: bool
    results: dict[str, Any]


# --- Helper Queries ---


def _get_active_inventory_output(session: Session) -> ActiveOutput | None:
    """Retrieve the current active output pointer for inventory recommendations."""
    return session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == TASK_INVENTORY)
    ).scalar_one_or_none()


def _calculate_params_synthetic_share(
    session: Session,
    active_inv: ActiveOutput | None,
) -> float:
    """Calculate the fraction of active recommendations that used synthetic parameters."""
    if active_inv is None:
        total_params = session.scalar(sa.select(sa.func.count()).select_from(InventoryParam)) or 0
        if total_params == 0:
            return 0.0
        synthetic_params = (
            session.scalar(
                sa.select(sa.func.count())
                .select_from(InventoryParam)
                .where(InventoryParam.source == "synthetic")
            )
            or 0
        )
        return round(synthetic_params / total_params, 4)

    total_recs = (
        session.scalar(
            sa.select(sa.func.count())
            .select_from(InventoryRecommendation)
            .where(
                InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
                InventoryRecommendation.as_of_date == active_inv.data_as_of,
            )
        )
        or 0
    )
    if total_recs == 0:
        return 0.0

    synthetic_recs = (
        session.scalar(
            sa.select(sa.func.count())
            .select_from(InventoryRecommendation)
            .where(
                InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
                InventoryRecommendation.as_of_date == active_inv.data_as_of,
                InventoryRecommendation.params_source == "synthetic",
            )
        )
        or 0
    )
    return round(synthetic_recs / total_recs, 4)


def _build_recommendation_item(
    rec: InventoryRecommendation,
    prod: DimProduct,
) -> InventoryRecommendationItem:
    """Map DB models to response item."""
    return InventoryRecommendationItem(
        product_id=rec.product_id,
        sku=prod.sku,
        description=prod.description,
        abc_class=prod.abc_class,
        store_id=rec.store_id,
        as_of_date=rec.as_of_date,
        on_hand=rec.on_hand,
        on_order=rec.on_order,
        lead_time_days=float(rec.lead_time_days),
        service_level=float(rec.service_level),
        demand_mean_lt=float(rec.demand_mean_lt) if rec.demand_mean_lt is not None else None,
        demand_std_lt=float(rec.demand_std_lt) if rec.demand_std_lt is not None else None,
        safety_stock=float(rec.safety_stock),
        reorder_point=float(rec.reorder_point),
        recommended_order_qty=int(rec.order_quantity),
        stockout_risk=float(rec.expected_stockout_prob),
        days_of_cover=float(rec.days_of_cover) if rec.days_of_cover is not None else None,
        status=rec.status,
        params_source=rec.params_source,
    )


# --- Endpoints ---


@router.get(
    "/inventory/summary",
    response_model=InventorySummaryResponse,
    operation_id="get_inventory_summary",
    summary="Get inventory summary metrics and risk distribution",
    dependencies=[Depends(require_role("viewer"))],
)
def get_inventory_summary(
    session: Annotated[Session, Depends(get_db)],
    store_id: Annotated[int | None, Query(description="Filter by store ID")] = None,
) -> InventorySummaryResponse:
    """Retrieve status counts, stockout risk distribution, and synthetic share."""
    active_inv = _get_active_inventory_output(session)
    default_counts = {
        "ok": 0,
        "reorder_now": 0,
        "overstock": 0,
        "stockout_risk": 0,
        "no_data": 0,
    }

    if active_inv is None:
        synthetic_share = _calculate_params_synthetic_share(session, None)
        return InventorySummaryResponse(
            status_counts=default_counts,
            stockout_risk_histogram=[],
            params_synthetic_share=synthetic_share,
        )

    # 1. Base query filter
    filters = [
        InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
        InventoryRecommendation.as_of_date == active_inv.data_as_of,
    ]
    if store_id is not None:
        filters.append(InventoryRecommendation.store_id == store_id)

    # 2. Status counts
    counts_stmt = (
        sa.select(InventoryRecommendation.status, sa.func.count(InventoryRecommendation.product_id))
        .where(*filters)
        .group_by(InventoryRecommendation.status)
    )
    status_counts = dict(default_counts)
    total_items = 0
    for st, count in session.execute(counts_stmt):
        status_counts[str(st)] = int(count)
        total_items += int(count)

    # 3. Stockout risk histogram (10 bins from 0.0 to 1.0)
    probs_stmt = sa.select(InventoryRecommendation.expected_stockout_prob).where(*filters)
    probs = session.execute(probs_stmt).scalars().all()

    bin_counts = [0] * 10
    for p in probs:
        val = float(p)
        idx = min(int(val * 10.0), 9)
        idx = max(0, idx)
        bin_counts[idx] += 1

    histogram = [
        HistogramBin(
            bin_start=round(i * 0.1, 1),
            bin_end=round((i + 1) * 0.1, 1),
            count=bin_counts[i],
        )
        for i in range(10)
    ]

    synthetic_share = _calculate_params_synthetic_share(session, active_inv)

    return InventorySummaryResponse(
        as_of_date=active_inv.data_as_of,
        total_items=total_items,
        status_counts=status_counts,
        stockout_risk_histogram=histogram,
        params_synthetic_share=synthetic_share,
    )


@router.get(
    "/inventory/recommendations",
    response_model=PaginatedInventoryRecommendationsResponse,
    operation_id="get_inventory_recommendations",
    summary="List paginated inventory replenishment recommendations",
    dependencies=[Depends(require_role("viewer"))],
)
def get_inventory_recommendations(  # noqa: PLR0917
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    status_filter: Annotated[
        str | None, Query(alias="status", description="Filter by status")
    ] = None,
    abc_class: Annotated[str | None, Query(description="Filter by ABC class (A, B, C)")] = None,
    q: Annotated[str | None, Query(description="Search SKU or description")] = None,
    store_id: Annotated[int | None, Query(description="Filter by store ID")] = None,
    sort: Annotated[
        str,
        Query(
            description=(
                "Sort expression (allowed: stockout_risk, days_of_cover, reorder_urgency, "
                "recommended_order_qty, product_id, sku). Default: stockout_risk:desc"
            )
        ),
    ] = "stockout_risk:desc",
) -> PaginatedInventoryRecommendationsResponse:
    """Retrieve paginated and filtered inventory recommendations."""
    sort_fields = parse_sort(
        sort,
        allowed_fields=RECOMMENDATION_SORT_WHITELIST.keys(),
        default="stockout_risk:desc",
    )

    active_inv = _get_active_inventory_output(session)
    if active_inv is None:
        synthetic_share = _calculate_params_synthetic_share(session, None)
        return PaginatedInventoryRecommendationsResponse(
            items=[],
            page=pagination.page,
            page_size=pagination.page_size,
            total=0,
            pages=0,
            params_synthetic_share=synthetic_share,
        )

    # 1. Base query joining DimProduct
    query: Any = (
        sa.select(InventoryRecommendation, DimProduct)
        .join(DimProduct, InventoryRecommendation.product_id == DimProduct.product_id)
        .where(
            InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
            InventoryRecommendation.as_of_date == active_inv.data_as_of,
        )
    )

    if store_id is not None:
        query = query.where(InventoryRecommendation.store_id == store_id)
    if status_filter:
        query = query.where(InventoryRecommendation.status == status_filter)
    if abc_class:
        query = query.where(DimProduct.abc_class == abc_class.upper())
    if q:
        search_pattern = f"%{q}%"
        query = query.where(
            DimProduct.sku.ilike(search_pattern) | DimProduct.description.ilike(search_pattern)
        )

    # 2. Count total
    count_stmt = sa.select(sa.func.count()).select_from(query.subquery())
    total = session.scalar(count_stmt) or 0

    # 3. Apply sorting and pagination
    query = apply_sort(query, sort_fields, RECOMMENDATION_SORT_WHITELIST)
    offset = (pagination.page - 1) * pagination.page_size
    query = query.offset(offset).limit(pagination.page_size)

    rows = session.execute(query).all()
    items: list[InventoryRecommendationItem] = []
    for row in rows:
        rec_obj: InventoryRecommendation = row[0]
        prod_obj: DimProduct = row[1]
        items.append(_build_recommendation_item(rec_obj, prod_obj))

    pages = (total + pagination.page_size - 1) // pagination.page_size if total > 0 else 0
    synthetic_share = _calculate_params_synthetic_share(session, active_inv)

    return PaginatedInventoryRecommendationsResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
        pages=pages,
        params_synthetic_share=synthetic_share,
    )


@router.get(
    "/inventory/recommendations/export",
    response_class=StreamingResponse,
    operation_id="export_inventory_recommendations",
    summary="Export filtered inventory recommendations as formula-safe CSV",
    dependencies=[Depends(require_role("analyst"))],
)
def export_inventory_recommendations(
    session: Annotated[Session, Depends(get_db)],
    status_filter: Annotated[
        str | None, Query(alias="status", description="Filter by status")
    ] = None,
    abc_class: Annotated[str | None, Query(description="Filter by ABC class (A, B, C)")] = None,
    q: Annotated[str | None, Query(description="Search SKU or description")] = None,
    store_id: Annotated[int | None, Query(description="Filter by store ID")] = None,
) -> StreamingResponse:
    """Export filtered inventory recommendations as a CSV attachment.

    Enforces SEC-INP-08 and SEC-LOG-01:
    - Neutralizes formula triggers with a single quote prefix.
    - Restricted to analyst+ role (viewers receive 403 Forbidden).
    """
    active_inv = _get_active_inventory_output(session)
    headers = [
        "product_id",
        "sku",
        "description",
        "abc_class",
        "store_id",
        "as_of_date",
        "on_hand",
        "on_order",
        "lead_time_days",
        "service_level",
        "demand_mean_lt",
        "demand_std_lt",
        "safety_stock",
        "reorder_point",
        "recommended_order_qty",
        "stockout_risk",
        "days_of_cover",
        "status",
        "params_source",
    ]

    if active_inv is None:
        return create_csv_streaming_response(
            rows=[],
            filename="inventory_recommendations.csv",
            headers=headers,
        )

    query = (
        sa.select(InventoryRecommendation, DimProduct)
        .join(DimProduct, InventoryRecommendation.product_id == DimProduct.product_id)
        .where(
            InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
            InventoryRecommendation.as_of_date == active_inv.data_as_of,
        )
    )

    if store_id is not None:
        query = query.where(InventoryRecommendation.store_id == store_id)
    if status_filter:
        query = query.where(InventoryRecommendation.status == status_filter)
    if abc_class:
        query = query.where(DimProduct.abc_class == abc_class.upper())
    if q:
        search_pattern = f"%{q}%"
        query = query.where(
            DimProduct.sku.ilike(search_pattern) | DimProduct.description.ilike(search_pattern)
        )

    query = query.order_by(InventoryRecommendation.expected_stockout_prob.desc())
    results = session.execute(query).all()

    export_rows = [
        [
            rec.product_id,
            prod.sku,
            prod.description or "",
            prod.abc_class or "",
            rec.store_id,
            rec.as_of_date.isoformat(),
            rec.on_hand,
            rec.on_order,
            float(rec.lead_time_days),
            float(rec.service_level),
            float(rec.demand_mean_lt) if rec.demand_mean_lt is not None else "",
            float(rec.demand_std_lt) if rec.demand_std_lt is not None else "",
            float(rec.safety_stock),
            float(rec.reorder_point),
            int(rec.order_quantity),
            float(rec.expected_stockout_prob),
            float(rec.days_of_cover) if rec.days_of_cover is not None else "",
            rec.status,
            rec.params_source,
        ]
        for rec, prod in results
    ]

    return create_csv_streaming_response(
        rows=export_rows,
        filename="inventory_recommendations.csv",
        headers=headers,
    )


@router.get(
    "/inventory/params/{product_id}",
    response_model=InventoryParamResponse,
    operation_id="get_inventory_params",
    summary="Get inventory control policy parameters for a product",
    dependencies=[Depends(require_role("viewer"))],
)
def get_inventory_params(
    product_id: int,
    session: Annotated[Session, Depends(get_db)],
    store_id: Annotated[int, Query(description="Store ID")] = 1,
) -> InventoryParamResponse:
    """Retrieve policy parameters for a specific product and store."""
    param = session.get(InventoryParam, (product_id, store_id))
    if param is None:
        raise NotFoundError(
            f"Inventory parameters not found for product_id={product_id}, store_id={store_id}"
        )

    return InventoryParamResponse(
        product_id=param.product_id,
        store_id=param.store_id,
        lead_time_days=float(param.lead_time_days),
        lead_time_std_days=float(param.lead_time_std_days),
        service_level=float(param.service_level),
        unit_cost=float(param.unit_cost) if param.unit_cost is not None else None,
        holding_cost_rate=float(param.holding_cost_rate),
        order_cost=float(param.order_cost),
        moq=param.moq,
        source=param.source,
        updated_at=param.updated_at,
    )


@router.put(
    "/inventory/params/{product_id}",
    response_model=InventoryParamResponse,
    operation_id="update_inventory_params",
    summary="Upsert inventory control parameters (source becomes 'user')",
)
def update_inventory_params(
    product_id: int,
    payload: UpdateInventoryParamRequest,
    current_user: Annotated[AppUser, Depends(require_role("analyst"))],
    session: Annotated[Session, Depends(get_db)],
    store_id: Annotated[int, Query(description="Store ID")] = 1,
) -> InventoryParamResponse:
    """Upsert inventory policy parameters, recording author and marking source as user."""
    # 1. Verify product exists
    prod = session.get(DimProduct, product_id)
    if prod is None:
        raise NotFoundError(f"Product {product_id} not found")

    # 2. Find or create InventoryParam
    param = session.get(InventoryParam, (product_id, store_id))
    now = utc_now()

    if param is None:
        param = InventoryParam(
            product_id=product_id,
            store_id=store_id,
            lead_time_days=Decimal(str(payload.lead_time_days)),
            lead_time_std_days=Decimal(str(payload.lead_time_std_days)),
            service_level=Decimal(str(payload.service_level)),
            unit_cost=Decimal(str(payload.unit_cost)) if payload.unit_cost is not None else None,
            holding_cost_rate=Decimal(str(payload.holding_cost_rate)),
            order_cost=Decimal(str(payload.order_cost)),
            moq=payload.moq,
            source="user",
            updated_by=current_user.user_id,
            updated_at=now,
        )
        session.add(param)
    else:
        param.lead_time_days = Decimal(str(payload.lead_time_days))
        param.lead_time_std_days = Decimal(str(payload.lead_time_std_days))
        param.service_level = Decimal(str(payload.service_level))
        param.unit_cost = Decimal(str(payload.unit_cost)) if payload.unit_cost is not None else None
        param.holding_cost_rate = Decimal(str(payload.holding_cost_rate))
        param.order_cost = Decimal(str(payload.order_cost))
        param.moq = payload.moq
        param.source = "user"
        param.updated_by = current_user.user_id
        param.updated_at = now

    log_audit(
        session=session,
        action="param_update",
        user_id=current_user.user_id,
        resource=f"inventory_param:{product_id}:{store_id}",
        details={
            "lead_time_days": float(payload.lead_time_days),
            "service_level": float(payload.service_level),
            "source": "user",
        },
    )

    session.commit()
    session.refresh(param)

    return InventoryParamResponse(
        product_id=param.product_id,
        store_id=param.store_id,
        lead_time_days=float(param.lead_time_days),
        lead_time_std_days=float(param.lead_time_std_days),
        service_level=float(param.service_level),
        unit_cost=float(param.unit_cost) if param.unit_cost is not None else None,
        holding_cost_rate=float(param.holding_cost_rate),
        order_cost=float(param.order_cost),
        moq=param.moq,
        source=param.source,
        updated_at=param.updated_at,
    )


@router.post(
    "/inventory/what-if",
    response_model=WhatIfResponse,
    operation_id="compute_inventory_what_if",
    summary="Simulate inventory policy changes with non-persistence guarantee",
    dependencies=[Depends(require_role("analyst"))],
)
def compute_inventory_what_if(  # noqa: PLR0915
    payload: WhatIfRequest,
    session: Annotated[Session, Depends(get_db)],
) -> WhatIfResponse:
    """Evaluate inventory policy overrides without saving anything to the database."""
    # 1. Product lookup
    prod = session.get(DimProduct, payload.product_id)
    if prod is None:
        raise NotFoundError(f"Product {payload.product_id} not found")

    overrides = payload.overrides

    # 2. Validate overrides
    if "lead_time_days" in overrides and float(overrides["lead_time_days"]) <= 0:
        raise DomainValidationError("lead_time_days must be greater than 0")
    if "service_level" in overrides:
        sl = float(overrides["service_level"])
        if not (0.50 < sl < 1.00):
            raise DomainValidationError("service_level must be between 0.50 and 1.00")
    if "moq" in overrides and int(overrides["moq"]) < 1:
        raise DomainValidationError("moq must be at least 1")
    if "holding_cost_rate" in overrides and float(overrides["holding_cost_rate"]) < 0:
        raise DomainValidationError("holding_cost_rate must be non-negative")
    if "order_cost" in overrides and float(overrides["order_cost"]) < 0:
        raise DomainValidationError("order_cost must be non-negative")

    # 3. Check for existing active recommendation
    active_inv = _get_active_inventory_output(session)
    rec: InventoryRecommendation | None = None
    if active_inv is not None:
        rec = session.execute(
            sa.select(InventoryRecommendation).where(
                InventoryRecommendation.forecast_run_id == active_inv.model_run_id,
                InventoryRecommendation.as_of_date == active_inv.data_as_of,
                InventoryRecommendation.product_id == payload.product_id,
                InventoryRecommendation.store_id == payload.store_id,
            )
        ).scalar_one_or_none()

    as_of = active_inv.data_as_of if active_inv else utc_now().date()

    if rec is not None:
        orig_lt = float(rec.lead_time_days)
        orig_mu_l = float(rec.demand_mean_lt or 0.0)
        mu_d = orig_mu_l / orig_lt if orig_lt > 0 else 0.0
        orig_sigma_l = float(rec.demand_std_lt or 0.0)
        sigma_e = orig_sigma_l / (orig_lt**0.5) if orig_lt > 0 else 1.0

        on_hand = int(overrides.get("on_hand", rec.on_hand))
        on_order = int(overrides.get("on_order", rec.on_order))
        lt_days = float(overrides.get("lead_time_days", orig_lt))
        lt_std_days = float(overrides.get("lead_time_std_days", 0.2 * lt_days))
        service_level = float(overrides.get("service_level", float(rec.service_level)))
        unit_cost = float(overrides.get("unit_cost", float(rec.unit_cost)))
        holding_cost_rate = float(overrides.get("holding_cost_rate", float(rec.holding_cost_rate)))
        order_cost = float(overrides.get("order_cost", float(rec.order_cost)))
        moq = int(overrides.get("moq", 1))
    else:
        # Fallback to InventoryParam, InventorySnapshot, or product defaults
        param = session.get(InventoryParam, (payload.product_id, payload.store_id))
        snap = session.execute(
            sa.select(InventorySnapshot)
            .where(
                InventorySnapshot.product_id == payload.product_id,
                InventorySnapshot.store_id == payload.store_id,
            )
            .order_by(InventorySnapshot.snapshot_date.desc())
        ).scalar_one_or_none()

        on_hand = int(overrides.get("on_hand", snap.on_hand if snap else 0))
        on_order = int(overrides.get("on_order", snap.on_order if snap else 0))

        if param is not None:
            lt_days = float(overrides.get("lead_time_days", float(param.lead_time_days)))
            lt_std_days = float(
                overrides.get("lead_time_std_days", float(param.lead_time_std_days))
            )
            service_level = float(overrides.get("service_level", float(param.service_level)))
            unit_cost = float(
                overrides.get(
                    "unit_cost",
                    float(param.unit_cost or prod.unit_cost_est or 10.0),
                )
            )
            holding_cost_rate = float(
                overrides.get("holding_cost_rate", float(param.holding_cost_rate))
            )
            order_cost = float(overrides.get("order_cost", float(param.order_cost)))
            moq = int(overrides.get("moq", param.moq))
        else:
            lt_days = float(overrides.get("lead_time_days", 14.0))
            lt_std_days = float(overrides.get("lead_time_std_days", 2.8))
            service_level = float(overrides.get("service_level", 0.95))
            unit_cost = float(overrides.get("unit_cost", float(prod.unit_cost_est or 10.0)))
            holding_cost_rate = float(overrides.get("holding_cost_rate", 0.20))
            order_cost = float(overrides.get("order_cost", 25.0))
            moq = int(overrides.get("moq", 1))

        mu_d = float(overrides.get("mu_d", 5.0))
        sigma_e = float(overrides.get("sigma_e", 2.0))

    is_eligible = bool(prod.is_product)

    # 4. Pure evaluation (no DB writes or commits)
    eval_res = evaluate_inventory_policy(
        on_hand=on_hand,
        on_order=on_order,
        lead_time_days=lt_days,
        lead_time_std_days=lt_std_days,
        mu_d=mu_d,
        sigma_e=sigma_e,
        unit_cost=unit_cost,
        holding_cost_rate=holding_cost_rate,
        order_cost=order_cost,
        service_level=service_level,
        moq=moq,
        is_eligible=is_eligible,
    )

    return WhatIfResponse(
        product_id=prod.product_id,
        sku=prod.sku,
        description=prod.description,
        abc_class=prod.abc_class,
        store_id=payload.store_id,
        as_of_date=as_of,
        on_hand=on_hand,
        on_order=on_order,
        lead_time_days=round(lt_days, 1),
        service_level=round(service_level, 3),
        demand_mean_lt=round(eval_res.mu_l, 1),
        demand_std_lt=round(eval_res.sigma_l, 1),
        safety_stock=round(eval_res.safety_stock, 1),
        reorder_point=round(eval_res.reorder_point, 1),
        recommended_order_qty=eval_res.recommended_order_qty,
        stockout_risk=round(eval_res.stockout_risk, 4),
        days_of_cover=round(eval_res.days_of_cover, 1) if eval_res.days_of_cover else None,
        status=eval_res.status,
        params_source="what_if",
        persisted=False,
    )


@router.post(
    "/inventory/simulate",
    response_model=SimulationResponse,
    operation_id="run_inventory_simulation",
    summary="Run synchronous policy simulation against historical demand",
    status_code=status.HTTP_200_OK,
)
@limiter.limit("20/hour")
def run_simulation(
    request: Request,
    payload: SimulationRequest,
    current_user: Annotated[AppUser, Depends(require_role("analyst"))],
    session: Annotated[Session, Depends(get_db)],
) -> SimulationResponse:
    """Run an inventory policy simulation (max 200 SKUs) and persist results."""
    del request
    # 1. Enforce max 200 SKUs limit (API §3.8: limit > 200 -> 422)
    if payload.scope.limit > MAX_SIMULATION_LIMIT:
        msg = (
            f"Simulation limit cannot exceed {MAX_SIMULATION_LIMIT} SKUs "
            f"(got {payload.scope.limit})"
        )
        raise DomainValidationError(msg)

    # 2. Build configuration
    cfg = SimulationConfig(
        abc_class=payload.scope.abc_class,
        limit=payload.scope.limit,
        service_level=payload.policy.service_level,
        baseline_type=payload.baseline.type,
        cover_days_extra=payload.baseline.cover_days_extra,
        persist=True,
    )

    # 3. Execute simulation
    sim_result = run_policy_simulation(
        session=session,
        config=cfg,
        created_by=current_user.user_id,
    )

    return SimulationResponse(
        simulation_id=sim_result.simulation_id,
        window=SimulationWindow(
            start=sim_result.window_start,
            end=sim_result.window_end,
        ),
        uses_synthetic_params=sim_result.uses_synthetic_params,
        results=sim_result.results,
    )


@router.get(
    "/inventory/simulations/latest",
    response_model=SimulationResponse,
    operation_id="get_latest_simulation",
    summary="Retrieve the most recent inventory policy simulation",
    dependencies=[Depends(require_role("viewer"))],
)
def get_latest_simulation(
    session: Annotated[Session, Depends(get_db)],
) -> SimulationResponse:
    """Retrieve the latest stored policy simulation run."""
    stmt = sa.select(PolicySimulation).order_by(PolicySimulation.created_at.desc()).limit(1)
    sim = session.execute(stmt).scalar_one_or_none()

    if sim is None:
        raise NotFoundError("No policy simulations have been recorded yet")

    return SimulationResponse(
        simulation_id=sim.simulation_id,
        window=SimulationWindow(
            start=sim.window_start,
            end=sim.window_end,
        ),
        uses_synthetic_params=sim.uses_synthetic_params,
        results=sim.results,
    )
