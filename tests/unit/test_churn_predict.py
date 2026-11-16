"""Unit and integration tests for customer churn batch scoring and active output management (P3-06).

Verifies:
- Identification of active customer population (>= 1 purchase in trailing 365 days).
- Model resolution (champion, specified run, or fallback).
- Probability prediction, risk band classification, and top-3 driver explanations.
- Risk bands strictly follow configured thresholds.
- Atomic persistence to ml.churn_score and ml.active_output.
- Transaction rollback on failure leaving previous active_output unchanged.
- Idempotency when scoring the same cutoff repeatedly.
- CLI command `retailpulse score churn`.
"""

from __future__ import annotations

import datetime
import uuid
from decimal import Decimal
from typing import Any

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from retailpulse.cli import app
from retailpulse.db.models.core import DimCustomer, DimDate, FactSalesLine
from retailpulse.db.models.ml import ActiveOutput, ChurnScore, ModelRun
from retailpulse.db.models.ops import IngestionBatch
from retailpulse.ml.churn.models import ALL_FEATURES
from retailpulse.ml.churn.predict import (
    TASK_CHURN,
    score_churn_pipeline,
)


@pytest.fixture
def in_memory_churn_db() -> Session:
    """Create in-memory SQLite database seeded with customers, dates, and sales lines."""
    engine = sa.create_engine("sqlite:///:memory:")
    engine = engine.execution_options(
        schema_translate_map={"core": None, "ops": None, "ml": None, "app": None}
    )

    for tbl in [
        DimCustomer.__table__,
        DimDate.__table__,
        IngestionBatch.__table__,
        FactSalesLine.__table__,
        ModelRun.__table__,
        ChurnScore.__table__,
        ActiveOutput.__table__,
    ]:
        tbl.create(engine, checkfirst=True)

    session_cls = sessionmaker(bind=engine)
    session = session_cls()

    # Seed customers
    customers = [
        DimCustomer(customer_id=1, customer_key="cust_1", country="United Kingdom"),
        DimCustomer(customer_id=2, customer_key="cust_2", country="France"),
        DimCustomer(customer_id=3, customer_key="cust_3", country="Germany"),
        DimCustomer(customer_id=4, customer_key="cust_4", country="United Kingdom"),
    ]
    session.add_all(customers)

    # Seed ingestion batch
    batch_id = uuid.uuid4()
    session.add(
        IngestionBatch(
            batch_id=batch_id,
            source_name="test",
            source_file="test.xlsx",
            file_sha256="a" * 64,
            status="succeeded",
        )
    )

    # Seed sales lines
    lines = [
        # Customer 1: frequent purchaser
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h1",
            invoice_no="INV1",
            invoice_ts=datetime.datetime(2011, 1, 15, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2011, 1, 15),
            customer_id=1,
            product_id=101,
            store_id=1,
            quantity=10,
            unit_price=Decimal("15.00"),
            is_return=False,
        ),
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h2",
            invoice_no="INV2",
            invoice_ts=datetime.datetime(2011, 6, 20, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2011, 6, 20),
            customer_id=1,
            product_id=102,
            store_id=1,
            quantity=5,
            unit_price=Decimal("20.00"),
            is_return=False,
        ),
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h3",
            invoice_no="INV3",
            invoice_ts=datetime.datetime(2011, 11, 10, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2011, 11, 10),
            customer_id=1,
            product_id=101,
            store_id=1,
            quantity=8,
            unit_price=Decimal("15.00"),
            is_return=False,
        ),
        # Customer 2: single order, older (at risk of churn)
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h4",
            invoice_no="INV4",
            invoice_ts=datetime.datetime(2011, 2, 1, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2011, 2, 1),
            customer_id=2,
            product_id=103,
            store_id=1,
            quantity=2,
            unit_price=Decimal("50.00"),
            is_return=False,
        ),
        # Customer 3: purchased recently
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h5",
            invoice_no="INV5",
            invoice_ts=datetime.datetime(2011, 11, 25, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2011, 11, 25),
            customer_id=3,
            product_id=104,
            store_id=1,
            quantity=12,
            unit_price=Decimal("10.00"),
            is_return=False,
        ),
        # Customer 4: purchased >365d ago (inactive, should be excluded from active scoring)
        FactSalesLine(
            batch_id=batch_id,
            source_row_hash="h6",
            invoice_no="INV6",
            invoice_ts=datetime.datetime(2010, 10, 1, 10, 0, tzinfo=datetime.UTC),
            date_key=datetime.date(2010, 10, 1),
            customer_id=4,
            product_id=105,
            store_id=1,
            quantity=1,
            unit_price=Decimal("30.00"),
            is_return=False,
        ),
    ]
    session.add_all(lines)
    session.commit()
    return session


def test_score_churn_pipeline_basic(in_memory_churn_db: Session) -> None:
    """Verify end-to-end churn scoring, bands, top drivers, and active pointer update."""
    session = in_memory_churn_db
    cutoff = datetime.date(2011, 12, 1)

    result = score_churn_pipeline(
        session=session,
        data_as_of=cutoff,
    )

    # Customers 1, 2, 3 are active in trailing 365 days; Customer 4 was > 365d
    assert result.n_customers == 3
    assert result.active_output_updated is True
    assert result.data_as_of == cutoff

    # Verify active output pointer
    active = session.get(ActiveOutput, TASK_CHURN)
    assert active is not None
    assert active.model_run_id == result.model_run_id
    assert active.data_as_of == cutoff

    # Verify rows in ml.churn_score
    scores = (
        session.execute(sa.select(ChurnScore).where(ChurnScore.model_run_id == result.model_run_id))
        .scalars()
        .all()
    )
    assert len(scores) == 3

    scored_cids = {s.customer_id for s in scores}
    assert scored_cids == {1, 2, 3}

    for s in scores:
        prob = float(s.churn_probability)
        assert 0.0 <= prob <= 1.0

        # Verify risk band rules
        if prob >= 0.60:
            assert s.risk_band == "high"
        elif prob >= 0.30:
            assert s.risk_band == "medium"
        else:
            assert s.risk_band == "low"

        # Verify top drivers (top 3)
        assert len(s.top_drivers) == 3
        for driver in s.top_drivers:
            assert set(driver.keys()) == {"feature", "value", "impact", "direction"}
            assert driver["feature"] in ALL_FEATURES
            assert driver["direction"] in {"increases_risk", "decreases_risk"}
            assert isinstance(driver["impact"], float)


def test_score_churn_pipeline_idempotent(in_memory_churn_db: Session) -> None:
    """Verify running score_churn_pipeline repeatedly cleanly overwrites scores."""
    session = in_memory_churn_db
    cutoff = datetime.date(2011, 12, 1)

    # First run
    res1 = score_churn_pipeline(session=session, data_as_of=cutoff)
    assert res1.n_customers == 3

    # Second run with same model_run_id
    res2 = score_churn_pipeline(
        session=session,
        data_as_of=cutoff,
        model_run_id=res1.model_run_id,
    )
    assert res2.n_customers == 3

    # Total records in churn_score for this model_run should still be exactly 3
    stmt = sa.select(sa.func.count(ChurnScore.customer_id)).where(
        ChurnScore.model_run_id == res1.model_run_id
    )
    count = session.execute(stmt).scalar()
    assert count == 3


def test_score_churn_pipeline_rollback_on_failure(
    in_memory_churn_db: Session, monkeypatch: Any
) -> None:
    """Verify atomic transaction rollback leaves active_output untouched if scoring fails."""
    session = in_memory_churn_db
    cutoff = datetime.date(2011, 12, 1)

    # Initial successful score
    _ = score_churn_pipeline(session=session, data_as_of=cutoff)
    orig_pointer = session.get(ActiveOutput, TASK_CHURN)
    assert orig_pointer is not None
    orig_run_id = orig_pointer.model_run_id

    # Simulate error during persistence by monkeypatching session.add_all to raise
    def fail_add_all(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Simulated DB failure")

    monkeypatch.setattr(session, "add_all", fail_add_all)

    new_cutoff = datetime.date(2011, 12, 5)
    with pytest.raises(RuntimeError, match="Simulated DB failure"):
        score_churn_pipeline(session=session, data_as_of=new_cutoff)

    # Check that previous active_output was preserved intact
    curr_pointer = session.get(ActiveOutput, TASK_CHURN)
    assert curr_pointer is not None
    assert curr_pointer.model_run_id == orig_run_id
    assert curr_pointer.data_as_of == cutoff


def test_score_churn_cli_help() -> None:
    """Verify CLI command options render correctly."""
    runner = CliRunner()
    result = runner.invoke(app, ["score", "churn", "--help"])
    assert result.exit_code == 0
    assert "--data-as-of" in result.stdout
    assert "--model-run-id" in result.stdout
    assert "--json" in result.stdout
