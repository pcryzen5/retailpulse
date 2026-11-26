"""Unit tests for query profiling pass, performance indexes, and benchmark reporting.

Implements verification for Task P9-01:
- Validates performance report generation and before/after table contents.
- Validates benchmark dataset seeding across core and ML tables.
- Validates SQLAlchemy model index registrations.
- Validates Alembic migration 0008 definitions for forward and reverse execution.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa
from scripts.profile_queries import (
    _run_profile,
    generate_performance_report,
    seed_benchmark_data,
)
from sqlalchemy.dialects.postgresql import CITEXT
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker

from retailpulse.db.base import Base
from retailpulse.db.models.core import DimCustomer, FactSalesLine
from retailpulse.db.models.ml import ChurnScore, Forecast, InventoryRecommendation

migration_0008 = importlib.import_module("migrations.versions.0008_performance_indexes")


@compiles(CITEXT, "sqlite")
def compile_citext_sqlite(type_, compiler, **kw):
    """Render PostgreSQL CITEXT as TEXT in SQLite for isolated tests."""
    return "TEXT"


@pytest.fixture
def profile_test_db():
    """In-memory SQLite database engine and session with translated schemas."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=sa.pool.StaticPool,
    )
    engine = engine.execution_options(
        schema_translate_map={"app": None, "core": None, "ops": None, "ml": None}
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    yield engine, session
    session.close()
    engine.dispose()


def test_seed_benchmark_data(profile_test_db: tuple[sa.Engine, Any]) -> None:
    """Verify that benchmark dataset seeds successfully across all required dimensions."""
    _, session = profile_test_db
    context = seed_benchmark_data(session)

    assert context["as_of"] is not None
    assert context["forecast_run_id"] is not None

    cust_count = session.scalar(sa.select(sa.func.count(DimCustomer.customer_id)))
    assert cust_count == 1000

    forecast_count = session.scalar(sa.select(sa.func.count(Forecast.horizon_day)))
    assert forecast_count == 150 * 28


def test_generate_performance_report(
    tmp_path: Path, profile_test_db: tuple[sa.Engine, Any]
) -> None:
    """Verify that report generation creates a markdown file with the required tables and plans."""
    engine, _ = profile_test_db
    results = _run_profile(engine, iterations=5)

    assert len(results) == 5
    out_file = tmp_path / "test_performance.md"
    content = generate_performance_report(results, out_file)

    assert out_file.exists()
    assert (
        "# RetailPulse — Query Profiling Pass & Performance Optimization Report (P9-01)"
        in content
    )
    assert "Before vs After Benchmark Comparison Table" in content
    assert "ix_dim_customer_last_order_date" in content
    assert "ix_fsl_date_agg" in content
    assert "ix_churn_score_run_date_band" in content
    assert "ix_inv_rec_run_date_status" in content
    assert "ix_forecast_run_prod_store_target" in content
    assert "EXPLAIN (ANALYZE, BUFFERS)" in content


def test_model_performance_index_registrations() -> None:
    """Verify that all 5 performance indexes are registered on SQLAlchemy model metadata."""
    # 1. DimCustomer
    dim_cust_indexes = {idx.name for idx in DimCustomer.__table__.indexes}
    assert "ix_dim_customer_last_order_date" in dim_cust_indexes

    # 2. FactSalesLine
    fsl_indexes = {idx.name for idx in FactSalesLine.__table__.indexes}
    assert "ix_fsl_date_agg" in fsl_indexes

    # 3. ChurnScore
    churn_indexes = {idx.name for idx in ChurnScore.__table__.indexes}
    assert "ix_churn_score_run_date_band" in churn_indexes

    # 4. InventoryRecommendation
    inv_indexes = {idx.name for idx in InventoryRecommendation.__table__.indexes}
    assert "ix_inv_rec_run_date_status" in inv_indexes

    # 5. Forecast
    forecast_indexes = {idx.name for idx in Forecast.__table__.indexes}
    assert "ix_forecast_run_prod_store_target" in forecast_indexes


def test_migration_0008_metadata() -> None:
    """Verify migration 0008 identifiers and dependency chain."""
    assert migration_0008.revision == "0008"
    assert migration_0008.down_revision == "0007"
    assert callable(migration_0008.upgrade)
    assert callable(migration_0008.downgrade)
