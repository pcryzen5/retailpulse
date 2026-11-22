"""Customer intelligence, segmentation, and churn risk API endpoints.

Implements API §2.4, §3.5, and SECURITY §1, §4.6:
- GET /api/v1/segments: Segment profiles for active segmentation.
- GET /api/v1/customers: Filtered and paginated customer list (Viewer+).
- GET /api/v1/customers/{customer_id}: Detailed customer profile with RFM and churn drivers.
- GET /api/v1/customers/export: Formula-safe CSV export (Analyst+ only).
- GET /api/v1/churn/summary: Risk band distributions, probability histogram, revenue at risk.
- Enforces customer privacy: raw source identifiers are never stored or exposed
  (only surrogate customer_id and pseudonymous customer_key).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import Annotated, Any

import sqlalchemy as sa
from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from starlette.responses import StreamingResponse

from retailpulse.api.dependencies import require_role
from retailpulse.api.pagination import (
    PaginatedResponse,
    PaginationParams,
    apply_sort,
    get_pagination_params,
    parse_sort,
)
from retailpulse.db.models.core import DimCustomer, FactSalesLine
from retailpulse.db.models.ml import (
    ActiveOutput,
    ChurnScore,
    CustomerSegment,
    ModelRun,
    SegmentProfile,
)
from retailpulse.db.session import get_db
from retailpulse.errors import DomainValidationError, NotFoundError
from retailpulse.security.csv_safety import create_csv_streaming_response
from retailpulse.security.pseudonymize import mask_customer_key

router = APIRouter(tags=["Customers & Churn"])


# --- Pydantic Schemas ---


class SegmentProfileResponse(BaseModel):
    """Segment profile summary model."""

    model_config = ConfigDict(extra="forbid")

    segment_id: int
    segment_label: str
    n_customers: int
    pct_customers: float
    avg_recency_days: float
    avg_frequency: float
    avg_monetary: Decimal
    revenue_share: float
    description: str | None = None


class CustomerListItem(BaseModel):
    """Customer summary item for list and table views."""

    model_config = ConfigDict(extra="forbid")

    customer_id: int
    customer_key: str
    country: str | None = None
    segment_id: int | None = None
    segment_label: str | None = None
    recency_days: int | None = None
    frequency: int | None = None
    monetary: Decimal | None = None
    churn_probability: float | None = None
    risk_band: str | None = None


class CustomerSegmentSummary(BaseModel):
    """Customer segment assignment header."""

    model_config = ConfigDict(extra="forbid")

    segment_id: int
    label: str


class CustomerRFMSummary(BaseModel):
    """Customer RFM behavioral metrics."""

    model_config = ConfigDict(extra="forbid")

    recency_days: int
    frequency: int
    monetary: float


class ChurnReasonItem(BaseModel):
    """Granular feature driver contributing to churn risk."""

    model_config = ConfigDict(extra="forbid")

    feature: str
    value: float | None = None
    impact: float
    direction: str


class CustomerChurnSummary(BaseModel):
    """Customer churn score breakdown."""

    model_config = ConfigDict(extra="forbid")

    probability: float
    risk_band: str
    as_of_date: datetime.date
    top_reasons: list[ChurnReasonItem]


class CustomerOrdersSummary(BaseModel):
    """Order history aggregates for a customer."""

    model_config = ConfigDict(extra="forbid")

    first_order_date: datetime.date | None = None
    last_order_date: datetime.date | None = None
    n_orders: int
    avg_order_value: float


class CustomerDetailResponse(BaseModel):
    """Detailed customer view matching API §3.5."""

    model_config = ConfigDict(extra="forbid")

    customer_id: int
    customer_key: str
    country: str | None = None
    segment: CustomerSegmentSummary | None = None
    rfm: CustomerRFMSummary | None = None
    churn: CustomerChurnSummary | None = None
    orders_summary: CustomerOrdersSummary


class HistogramBin(BaseModel):
    """Single bucket in churn probability histogram."""

    model_config = ConfigDict(extra="forbid")

    bin_start: float
    bin_end: float
    count: int


class ChurnSummaryResponse(BaseModel):
    """Global churn analytics summary."""

    model_config = ConfigDict(extra="forbid")

    as_of_date: datetime.date | None = None
    total_scored: int = 0
    band_counts: dict[str, int]
    probability_histogram: list[HistogramBin]
    mean_churn_probability: float = 0.0
    expected_revenue_at_risk: Decimal = Decimal("0.00")


# --- Query Helpers ---


def _build_customers_query(
    session: Session,
    segment_id: int | None = None,
    risk_band: str | None = None,
    min_probability: float | None = None,
    q: str | None = None,
) -> tuple[Any, ActiveOutput | None, ActiveOutput | None]:
    """Construct joined query for customers with active segmentation and churn."""
    active_seg = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "segmentation")
    ).scalar_one_or_none()

    active_churn = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "churn")
    ).scalar_one_or_none()

    query = sa.select(
        DimCustomer.customer_id,
        DimCustomer.customer_key,
        DimCustomer.country,
        CustomerSegment.segment_id,
        SegmentProfile.segment_label,
        CustomerSegment.recency_days,
        CustomerSegment.frequency,
        CustomerSegment.monetary,
        ChurnScore.churn_probability,
        ChurnScore.risk_band,
    )

    # Left join active segmentation
    if active_seg is not None:
        seg_join = sa.and_(
            CustomerSegment.customer_id == DimCustomer.customer_id,
            CustomerSegment.model_run_id == active_seg.model_run_id,
            CustomerSegment.as_of_date == active_seg.data_as_of,
        )
        query = query.outerjoin(CustomerSegment, seg_join)

        prof_join = sa.and_(
            SegmentProfile.segment_id == CustomerSegment.segment_id,
            SegmentProfile.model_run_id == active_seg.model_run_id,
            SegmentProfile.as_of_date == active_seg.data_as_of,
        )
        query = query.outerjoin(SegmentProfile, prof_join)
    else:
        query = query.outerjoin(CustomerSegment, sa.sql.false()).outerjoin(
            SegmentProfile, sa.sql.false()
        )

    # Left join active churn
    if active_churn is not None:
        churn_join = sa.and_(
            ChurnScore.customer_id == DimCustomer.customer_id,
            ChurnScore.model_run_id == active_churn.model_run_id,
            ChurnScore.as_of_date == active_churn.data_as_of,
        )
        query = query.outerjoin(ChurnScore, churn_join)
    else:
        query = query.outerjoin(ChurnScore, sa.sql.false())

    # Apply filters
    if segment_id is not None:
        query = query.where(CustomerSegment.segment_id == segment_id)

    if risk_band is not None:
        clean_band = risk_band.strip().lower()
        if clean_band not in ("low", "medium", "high"):
            raise DomainValidationError("risk_band must be one of 'low', 'medium', or 'high'")
        query = query.where(ChurnScore.risk_band == clean_band)

    if min_probability is not None:
        if not 0.0 <= min_probability <= 1.0:
            raise DomainValidationError("min_probability must be between 0.0 and 1.0")
        query = query.where(ChurnScore.churn_probability >= min_probability)

    if q:
        term = f"{q.strip()}%"
        query = query.where(DimCustomer.customer_key.ilike(term))

    return query, active_seg, active_churn


# --- Endpoints ---


@router.get(
    "/segments",
    summary="List Customer Segments",
    response_model=list[SegmentProfileResponse],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def list_segments(
    session: Annotated[Session, Depends(get_db)],
) -> list[SegmentProfileResponse]:
    """Retrieve segment profiles and personas for the active customer segmentation."""
    active_seg = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "segmentation")
    ).scalar_one_or_none()

    if active_seg is None:
        return []

    profiles = (
        session.execute(
            sa.select(SegmentProfile)
            .where(
                SegmentProfile.model_run_id == active_seg.model_run_id,
                SegmentProfile.as_of_date == active_seg.data_as_of,
            )
            .order_by(SegmentProfile.segment_id.asc())
        )
        .scalars()
        .all()
    )

    total_custs = sum(p.n_customers for p in profiles) or 1
    return [
        SegmentProfileResponse(
            segment_id=p.segment_id,
            segment_label=p.segment_label,
            n_customers=p.n_customers,
            pct_customers=round(float(p.n_customers) / float(total_custs), 4),
            avg_recency_days=float(p.avg_recency_days),
            avg_frequency=float(p.avg_frequency),
            avg_monetary=Decimal(str(p.avg_monetary)),
            revenue_share=float(p.revenue_share),
            description=None,
        )
        for p in profiles
    ]


@router.get(
    "/customers",
    summary="List Customers",
    response_model=PaginatedResponse[CustomerListItem],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def list_customers(  # noqa: PLR0917
    session: Annotated[Session, Depends(get_db)],
    pagination: Annotated[PaginationParams, Depends(get_pagination_params)],
    segment_id: Annotated[int | None, Query(description="Filter by segment ID")] = None,
    risk_band: Annotated[
        str | None, Query(description="Filter by risk band ('low', 'medium', 'high')")
    ] = None,
    min_probability: Annotated[
        float | None, Query(description="Minimum churn probability (0.0 to 1.0)")
    ] = None,
    q: Annotated[str | None, Query(description="Prefix match on customer_key")] = None,
    sort: Annotated[
        str | None,
        Query(
            description=(
                "Sort clause (e.g. 'churn_probability:desc', 'monetary:asc', 'recency_days:desc')"
            )
        ),
    ] = None,
) -> PaginatedResponse[CustomerListItem]:
    """List and search pseudonymous customer profiles with RFM and churn indicators."""
    query, _, _ = _build_customers_query(
        session=session,
        segment_id=segment_id,
        risk_band=risk_band,
        min_probability=min_probability,
        q=q,
    )

    allowed_sort_fields = {
        "customer_id": DimCustomer.customer_id,
        "churn_probability": ChurnScore.churn_probability,
        "monetary": CustomerSegment.monetary,
        "recency_days": CustomerSegment.recency_days,
        "frequency": CustomerSegment.frequency,
    }

    sort_clauses = parse_sort(
        sort,
        allowed_fields=allowed_sort_fields.keys(),
        default="customer_id:asc",
    )
    query = apply_sort(query, sort_clauses, allowed_sort_fields)

    # Execute count
    count_subquery = query.order_by(None).subquery()
    count_query = sa.select(sa.func.count()).select_from(count_subquery)
    total = session.scalar(count_query) or 0

    # Paginate
    paged_query = query.offset(pagination.offset).limit(pagination.limit)
    rows = session.execute(paged_query).all()

    items = [
        CustomerListItem(
            customer_id=r.customer_id,
            customer_key=mask_customer_key(r.customer_key),
            country=r.country,
            segment_id=r.segment_id,
            segment_label=r.segment_label,
            recency_days=r.recency_days,
            frequency=r.frequency,
            monetary=Decimal(str(r.monetary)) if r.monetary is not None else None,
            churn_probability=float(r.churn_probability)
            if r.churn_probability is not None
            else None,
            risk_band=r.risk_band,
        )
        for r in rows
    ]

    return PaginatedResponse(
        items=items,
        page=pagination.page,
        page_size=pagination.page_size,
        total=total,
    )


@router.get(
    "/customers/export",
    summary="Export Filtered Customers to CSV",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("analyst"))],
)
def export_customers(  # noqa: PLR0917
    session: Annotated[Session, Depends(get_db)],
    segment_id: Annotated[int | None, Query(description="Filter by segment ID")] = None,
    risk_band: Annotated[
        str | None, Query(description="Filter by risk band ('low', 'medium', 'high')")
    ] = None,
    min_probability: Annotated[
        float | None, Query(description="Minimum churn probability (0.0 to 1.0)")
    ] = None,
    q: Annotated[str | None, Query(description="Prefix match on customer_key")] = None,
    sort: Annotated[str | None, Query(description="Sort clause")] = None,
) -> StreamingResponse:
    """Export filtered customer list to CSV with formula-injection neutralization (Analyst+)."""
    query, _, _ = _build_customers_query(
        session=session,
        segment_id=segment_id,
        risk_band=risk_band,
        min_probability=min_probability,
        q=q,
    )

    allowed_sort_fields = {
        "customer_id": DimCustomer.customer_id,
        "churn_probability": ChurnScore.churn_probability,
        "monetary": CustomerSegment.monetary,
        "recency_days": CustomerSegment.recency_days,
        "frequency": CustomerSegment.frequency,
    }

    sort_clauses = parse_sort(
        sort,
        allowed_fields=allowed_sort_fields.keys(),
        default="customer_id:asc",
    )
    query = apply_sort(query, sort_clauses, allowed_sort_fields)

    # Stream or batch up to export limit
    rows = session.execute(query.limit(10000)).all()

    headers = [
        "customer_id",
        "customer_key",
        "country",
        "segment_id",
        "segment_label",
        "recency_days",
        "frequency",
        "monetary",
        "churn_probability",
        "risk_band",
    ]

    data_rows = [
        [
            r.customer_id,
            r.customer_key,  # Pseudonymous salted key
            r.country or "",
            r.segment_id or "",
            r.segment_label or "",
            r.recency_days or "",
            r.frequency or "",
            float(r.monetary) if r.monetary is not None else "",
            float(r.churn_probability) if r.churn_probability is not None else "",
            r.risk_band or "",
        ]
        for r in rows
    ]

    return create_csv_streaming_response(
        rows=data_rows,
        filename="customers_export.csv",
        headers=headers,
    )


@router.get(
    "/customers/{customer_id}",
    summary="Get Customer Detail",
    response_model=CustomerDetailResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_customer(
    customer_id: int,
    session: Annotated[Session, Depends(get_db)],
) -> CustomerDetailResponse:
    """Retrieve full customer profile, segment, RFM, and churn risk drivers."""
    customer = session.execute(
        sa.select(DimCustomer).where(DimCustomer.customer_id == customer_id)
    ).scalar_one_or_none()

    if customer is None:
        raise NotFoundError(f"Customer {customer_id} not found")

    # 1. Segment profile
    segment_summary: CustomerSegmentSummary | None = None
    rfm_summary: CustomerRFMSummary | None = None

    active_seg = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "segmentation")
    ).scalar_one_or_none()

    if active_seg is not None:
        seg_record = session.execute(
            sa.select(CustomerSegment, SegmentProfile.segment_label)
            .outerjoin(
                SegmentProfile,
                sa.and_(
                    SegmentProfile.segment_id == CustomerSegment.segment_id,
                    SegmentProfile.model_run_id == active_seg.model_run_id,
                    SegmentProfile.as_of_date == active_seg.data_as_of,
                ),
            )
            .where(
                CustomerSegment.customer_id == customer_id,
                CustomerSegment.model_run_id == active_seg.model_run_id,
                CustomerSegment.as_of_date == active_seg.data_as_of,
            )
        ).first()

        if seg_record is not None:
            c_seg, seg_label = seg_record
            segment_summary = CustomerSegmentSummary(
                segment_id=c_seg.segment_id,
                label=seg_label or f"Segment {c_seg.segment_id}",
            )
            rfm_summary = CustomerRFMSummary(
                recency_days=c_seg.recency_days,
                frequency=c_seg.frequency,
                monetary=float(c_seg.monetary),
            )

    # 2. Churn score
    churn_summary: CustomerChurnSummary | None = None
    active_churn = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "churn")
    ).scalar_one_or_none()

    if active_churn is not None:
        score_record = session.execute(
            sa.select(ChurnScore).where(
                ChurnScore.customer_id == customer_id,
                ChurnScore.model_run_id == active_churn.model_run_id,
                ChurnScore.as_of_date == active_churn.data_as_of,
            )
        ).scalar_one_or_none()

        if score_record is not None:
            drivers = [
                ChurnReasonItem(
                    feature=d.get("feature", "unknown"),
                    value=d.get("value"),
                    impact=float(d.get("impact", 0.0)),
                    direction=d.get("direction", "increases_risk"),
                )
                for d in score_record.top_drivers
            ]
            churn_summary = CustomerChurnSummary(
                probability=float(score_record.churn_probability),
                risk_band=score_record.risk_band,
                as_of_date=score_record.as_of_date,
                top_reasons=drivers,
            )

    # 3. Orders history summary
    order_stats = session.execute(
        sa.select(
            sa.func.count(sa.distinct(FactSalesLine.invoice_no)).label("n_orders"),
            sa.func.sum(FactSalesLine.line_amount).label("total_spend"),
            sa.func.min(FactSalesLine.date_key).label("first_date"),
            sa.func.max(FactSalesLine.date_key).label("last_date"),
        ).where(
            FactSalesLine.customer_id == customer_id,
            FactSalesLine.is_return.is_(False),
        )
    ).first()

    n_orders = int(order_stats.n_orders or 0) if order_stats else 0
    total_spend = Decimal(str(order_stats.total_spend or 0)) if order_stats else Decimal("0.00")
    avg_order_val = float(round(total_spend / n_orders, 2)) if n_orders > 0 else 0.0
    first_order_d = (
        order_stats.first_date
        if order_stats and order_stats.first_date
        else customer.first_order_date
    )
    last_order_d = (
        order_stats.last_date if order_stats and order_stats.last_date else customer.last_order_date
    )

    orders_summary = CustomerOrdersSummary(
        first_order_date=first_order_d,
        last_order_date=last_order_d,
        n_orders=n_orders,
        avg_order_value=avg_order_val,
    )

    return CustomerDetailResponse(
        customer_id=customer.customer_id,
        customer_key=mask_customer_key(customer.customer_key),
        country=customer.country,
        segment=segment_summary,
        rfm=rfm_summary,
        churn=churn_summary,
        orders_summary=orders_summary,
    )


@router.get(
    "/churn/summary",
    summary="Churn Population Summary",
    response_model=ChurnSummaryResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_role("viewer"))],
)
def get_churn_summary(
    session: Annotated[Session, Depends(get_db)],
) -> ChurnSummaryResponse:
    """Retrieve aggregate churn statistics, band counts, and probability histogram."""
    active_churn = session.execute(
        sa.select(ActiveOutput).where(ActiveOutput.task == "churn")
    ).scalar_one_or_none()

    if active_churn is None:
        return ChurnSummaryResponse(
            band_counts={"low": 0, "medium": 0, "high": 0},
            probability_histogram=[],
        )

    # 1. Band counts
    band_records = session.execute(
        sa.select(ChurnScore.risk_band, sa.func.count(ChurnScore.customer_id))
        .where(
            ChurnScore.model_run_id == active_churn.model_run_id,
            ChurnScore.as_of_date == active_churn.data_as_of,
        )
        .group_by(ChurnScore.risk_band)
    ).all()

    band_counts = {"low": 0, "medium": 0, "high": 0}
    total_scored = 0
    for band, count in band_records:
        band_counts[str(band)] = int(count)
        total_scored += int(count)

    # 2. Probability statistics & histogram (10 bins from 0.0 to 1.0)
    all_probs = (
        session.execute(
            sa.select(ChurnScore.churn_probability).where(
                ChurnScore.model_run_id == active_churn.model_run_id,
                ChurnScore.as_of_date == active_churn.data_as_of,
            )
        )
        .scalars()
        .all()
    )

    probs_float = [float(p) for p in all_probs]
    mean_prob = float(sum(probs_float) / len(probs_float)) if probs_float else 0.0

    bins: list[HistogramBin] = []
    for i in range(10):
        start = round(i * 0.1, 1)
        end = round((i + 1) * 0.1, 1)
        if i == 9:
            c = sum(1 for p in probs_float if start <= p <= 1.0)
        else:
            c = sum(1 for p in probs_float if start <= p < end)
        bins.append(HistogramBin(bin_start=start, bin_end=end, count=c))

    # 3. Expected revenue at risk
    churn_run = session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == active_churn.model_run_id)
    ).scalar_one_or_none()

    rev_at_risk = Decimal("0.00")
    if churn_run is not None and churn_run.metrics:
        raw_val = churn_run.metrics.get(
            "expected_revenue_at_risk",
            churn_run.metrics.get("revenue_at_risk", 0.0),
        )
        rev_at_risk = Decimal(str(raw_val))

    return ChurnSummaryResponse(
        as_of_date=active_churn.data_as_of,
        total_scored=total_scored,
        band_counts=band_counts,
        probability_histogram=bins,
        mean_churn_probability=round(mean_prob, 4),
        expected_revenue_at_risk=rev_at_risk,
    )
