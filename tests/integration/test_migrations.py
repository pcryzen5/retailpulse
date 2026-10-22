"""Integration and schema tests for Alembic migrations and SQLAlchemy models.

Covers:
  - Alembic forward migration (upgrade head) and reverse migration (downgrade base)
  - SQLAlchemy model registry, table definitions, schemas, and constraint naming conventions
  - Live PostgreSQL up/down/up cycle when database is available
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import TextIO

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Engine

# Ensure all models are registered
import retailpulse.db.models  # noqa: F401
from retailpulse.config import get_settings
from retailpulse.db.base import NAMING_CONVENTION, Base

ALEMBIC_INI = Path("alembic.ini")


def get_alembic_config(stdout: TextIO | None = None) -> Config:
    """Build Alembic Config object with project root settings."""
    cfg = Config(str(ALEMBIC_INI), stdout=stdout if stdout is not None else sys.stdout)
    cfg.set_main_option("script_location", "migrations")
    return cfg


def test_models_metadata_conventions() -> None:
    """Verify all expected tables are defined with the required schema and naming convention."""
    assert Base.metadata.naming_convention == NAMING_CONVENTION

    expected_tables = {
        "app.app_user",
        "ops.ingestion_batch",
        "ops.pipeline_run",
        "ops.data_quality_result",
        "core.dim_store",
        "core.dim_product",
        "core.dim_customer",
        "core.dim_date",
        "core.fact_sales_line",
        "core.fact_daily_sales",
        "core.inventory_param",
    }

    actual_tables = set(Base.metadata.tables.keys())
    assert expected_tables.issubset(actual_tables), (
        f"Missing tables in metadata: {expected_tables - actual_tables}"
    )

    # Verify specific constraints and conventions
    user_table = Base.metadata.tables["app.app_user"]
    assert user_table.schema == "app"
    assert "email" in user_table.c
    assert any(
        "admin" in str(ck.sqltext)
        for ck in user_table.constraints
        if isinstance(ck, sa.CheckConstraint)
    )

    sales_line = Base.metadata.tables["core.fact_sales_line"]
    assert sales_line.schema == "core"
    assert "source_row_hash" in sales_line.c
    assert any(
        "is_return" in str(ck.sqltext)
        for ck in sales_line.constraints
        if isinstance(ck, sa.CheckConstraint)
    )


def test_migrations_sql_generation() -> None:
    """Verify that Alembic generates complete SQL for all revisions up to head, and down to base."""
    # 1. Forward migration upgrade head SQL
    out_up = io.StringIO()
    cfg_up = get_alembic_config(stdout=out_up)
    with contextlib.redirect_stdout(out_up):
        command.upgrade(cfg_up, "head", sql=True)
    sql_up = out_up.getvalue()

    assert "CREATE TABLE app.app_user" in sql_up
    assert "CREATE TABLE ops.ingestion_batch" in sql_up
    assert "CREATE TABLE ops.pipeline_run" in sql_up
    assert "CREATE TABLE ops.data_quality_result" in sql_up
    assert "CREATE TABLE core.dim_store" in sql_up
    assert "CREATE TABLE core.dim_product" in sql_up
    assert "CREATE TABLE core.fact_sales_line" in sql_up
    assert "CREATE TABLE core.fact_daily_sales" in sql_up
    assert "CREATE TABLE core.inventory_param" in sql_up
    assert "INSERT INTO core.dim_store" in sql_up

    # 2. Reverse migration downgrade base SQL
    out_down = io.StringIO()
    cfg_down = get_alembic_config(stdout=out_down)
    with contextlib.redirect_stdout(out_down):
        command.downgrade(cfg_down, "0004:base", sql=True)
    sql_down = out_down.getvalue()

    assert "DROP TABLE core.inventory_param" in sql_down
    assert "DROP TABLE core.fact_daily_sales" in sql_down
    assert "DROP TABLE core.fact_sales_line" in sql_down
    assert "DROP TABLE ops.data_quality_result" in sql_down
    assert "DROP TABLE ops.pipeline_run" in sql_down
    assert "DROP TABLE ops.ingestion_batch" in sql_down
    assert "DROP TABLE app.app_user" in sql_down
    assert "DROP SCHEMA IF EXISTS core CASCADE" in sql_down


@pytest.mark.integration
def test_live_migrations_up_down_cycle() -> None:
    """Execute upgrade -> downgrade -> upgrade against a live PostgreSQL instance when available."""
    settings = get_settings()
    engine: Engine | None = None
    try:
        engine = sa.create_engine(
            settings.database_url,
            connect_args={"connect_timeout": 1},
            pool_pre_ping=False,
        )
        with engine.connect():
            pass
    except Exception as exc:
        pytest.skip(
            f"Live PostgreSQL instance not reachable ({exc}); skipping live migration cycle."
        )

    assert engine is not None
    cfg = get_alembic_config()

    try:
        # Full upgrade to head
        command.upgrade(cfg, "head")

        # Full downgrade to base
        command.downgrade(cfg, "base")

        # Re-upgrade to head to prove idempotent repeatability
        command.upgrade(cfg, "head")
    finally:
        engine.dispose()
