"""Integration tests for MLflow tracking, model registry, and database mirroring.

Verifies P2-09 & TRD §6:
- Parameter, metric, tag, and artifact logging in MLflow
- Standard tags: git_sha, dataset_hash, data_as_of, seed, library_versions
- Database mirroring to ml.model_run and ml.forecast_metric
- Model registration in MLflow registry
- Promotion and rollback service functions: champion and previous_champion aliases
- Mirroring of is_champion status in ml.model_run
- VERIFY REQUIREMENT: Integration test with a temporary file-store MLflow
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import sqlalchemy as sa
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session, sessionmaker

from retailpulse.db.models.core import DimCustomer, DimDate, DimProduct, DimStore
from retailpulse.db.models.ml import ForecastMetric, ModelRun
from retailpulse.ml.registry import (
    ALIAS_CHAMPION,
    ALIAS_PREVIOUS_CHAMPION,
    REGISTERED_MODEL_FORECAST,
    get_champion_run,
    get_champion_version,
    promote_champion,
    register_model,
    rollback_champion,
)
from retailpulse.ml.runs import (
    EXPERIMENT_FORECAST,
    get_git_sha,
    get_library_versions,
    track_model_run,
)


@pytest.fixture
def temp_mlflow_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create an isolated file-store tracking URI."""
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow_dir = tmp_path / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    return mlflow_dir.as_uri()


@pytest.fixture
def ml_db_session() -> Session:
    """In-memory SQLite session with schema translation map for testing."""
    engine = sa.create_engine(
        "sqlite:///:memory:",
        execution_options={
            "schema_translate_map": {"core": None, "ops": None, "ml": None, "app": None}
        },
    )
    tables = [
        cast(sa.Table, DimStore.__table__),
        cast(sa.Table, DimProduct.__table__),
        cast(sa.Table, DimCustomer.__table__),
        cast(sa.Table, DimDate.__table__),
        cast(sa.Table, ModelRun.__table__),
        cast(sa.Table, ForecastMetric.__table__),
    ]
    for table in tables:
        table.create(engine)

    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    return session


def test_git_sha_and_library_versions_helpers() -> None:
    """Verify system tags extraction produces non-empty strings."""
    sha = get_git_sha()
    assert isinstance(sha, str)
    assert len(sha) > 0

    lib_versions = get_library_versions()
    assert "mlflow=" in lib_versions
    assert "lightgbm=" in lib_versions
    assert "torch=" in lib_versions


def test_track_model_run_and_db_mirroring(
    temp_mlflow_uri: str,
    ml_db_session: Session,
    tmp_path: Path,
) -> None:
    """Verify logging of params, metrics, tags, artifacts, and mirroring to ml.model_run."""
    artifact_file = tmp_path / "summary.txt"
    artifact_file.write_text("backtest completed successfully")

    as_of = datetime.date(2011, 4, 1)

    with track_model_run(
        task="forecast",
        algorithm="lightgbm",
        data_as_of=as_of,
        dataset_hash="hash_123456",
        seed=42,
        session=ml_db_session,
        tracking_uri=temp_mlflow_uri,
        experiment_name=EXPERIMENT_FORECAST,
    ) as tracker:
        tracker.log_param("objective", "tweedie")
        tracker.log_param("n_estimators", 50)
        tracker.log_metric("global_wape", 0.1245)
        tracker.log_metric("global_bias", 0.0150)
        tracker.log_artifact(str(artifact_file))
        tracker.log_dict({"gate_passed": True, "reason": "better_than_baseline"}, "gate.json")

        # Mirror forecast metrics
        metrics_df = pd.DataFrame(
            [
                {
                    "scope": "global",
                    "scope_key": "all",
                    "metric": "wape",
                    "value": 0.1245,
                    "n_series": 30,
                },
                {
                    "scope": "abc",
                    "scope_key": "A",
                    "metric": "wape",
                    "value": 0.1050,
                    "n_series": 10,
                },
            ]
        )
        tracker.mirror_forecast_metrics(metrics_df, session=ml_db_session)

    # 1. Verify MLflow records
    client = MlflowClient(tracking_uri=temp_mlflow_uri)
    run = client.get_run(tracker.mlflow_run_id)

    assert run.info.status == "FINISHED"
    assert run.data.params["objective"] == "tweedie"
    assert run.data.params["n_estimators"] == "50"
    assert run.data.metrics["global_wape"] == pytest.approx(0.1245)
    assert run.data.tags["git_sha"] != ""
    assert run.data.tags["dataset_hash"] == "hash_123456"
    assert run.data.tags["data_as_of"] == str(as_of)

    # 2. Verify Database records in ml.model_run
    db_run = ml_db_session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == tracker.run_id)
    ).scalar_one()

    assert db_run.task == "forecast"
    assert db_run.algorithm == "lightgbm"
    assert db_run.mlflow_run_id == tracker.mlflow_run_id
    assert db_run.status == "succeeded"
    assert db_run.is_champion is False
    assert db_run.params["objective"] == "tweedie"
    assert db_run.metrics["global_wape"] == pytest.approx(0.1245)

    # 3. Verify Database records in ml.forecast_metric
    db_metrics = (
        ml_db_session.execute(
            sa.select(ForecastMetric).where(ForecastMetric.model_run_id == tracker.run_id)
        )
        .scalars()
        .all()
    )

    assert len(db_metrics) == 2
    scopes = {m.scope: m.value for m in db_metrics}
    assert scopes["global"] == pytest.approx(0.1245)
    assert scopes["abc"] == pytest.approx(0.1050)


def test_model_run_failure_handling(
    temp_mlflow_uri: str,
    ml_db_session: Session,
) -> None:
    """Verify that exceptions mark run as failed in both MLflow and database."""
    run_id: str | None = None
    app_id: uuid.UUID | None = None

    def _execute_failing_run() -> None:
        nonlocal run_id, app_id
        with track_model_run(
            task="forecast",
            algorithm="lstm",
            session=ml_db_session,
            tracking_uri=temp_mlflow_uri,
        ) as tracker:
            run_id = tracker.mlflow_run_id
            app_id = tracker.run_id
            tracker.log_param("hidden_size", 64)
            raise RuntimeError("Deliberate training failure")

    with pytest.raises(RuntimeError, match="Deliberate training failure"):
        _execute_failing_run()

    assert run_id is not None
    assert app_id is not None
    client = MlflowClient(tracking_uri=temp_mlflow_uri)
    run = client.get_run(run_id)
    assert run.info.status == "FAILED"

    db_run = ml_db_session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == app_id)
    ).scalar_one()
    assert db_run.status == "failed"


def test_model_registry_promote_and_rollback(
    temp_mlflow_uri: str,
    ml_db_session: Session,
    tmp_path: Path,
) -> None:
    """Verify model registration, champion promotion, alias updates, and rollback."""
    # Create fake model artifact
    dummy_model_file = tmp_path / "model.pt"
    dummy_model_file.write_text("model weights")

    # Run 1
    with track_model_run(
        task="forecast",
        algorithm="seasonal_naive",
        session=ml_db_session,
        tracking_uri=temp_mlflow_uri,
    ) as tracker1:
        tracker1.log_artifact(str(dummy_model_file), artifact_path="model")
        tracker1.log_metric("wape", 0.20)

    # Register Run 1
    ver1 = register_model(
        mlflow_run_id=tracker1.mlflow_run_id,
        task="forecast",
        artifact_path="model",
        tracking_uri=temp_mlflow_uri,
    )
    assert ver1 == "1"

    # Promote Run 1 to champion
    promo1 = promote_champion(
        task="forecast",
        version=ver1,
        session=ml_db_session,
        tracking_uri=temp_mlflow_uri,
    )
    assert promo1["champion_version"] == "1"
    assert promo1["previous_champion_version"] is None

    # Check champion in MLflow and DB
    assert get_champion_version("forecast", tracking_uri=temp_mlflow_uri) == "1"
    champ_run = get_champion_run("forecast", session=ml_db_session)
    assert champ_run is not None
    assert champ_run.run_id == tracker1.run_id
    assert champ_run.is_champion is True

    # Run 2 (Challenger beats champion)
    with track_model_run(
        task="forecast",
        algorithm="lightgbm",
        session=ml_db_session,
        tracking_uri=temp_mlflow_uri,
    ) as tracker2:
        tracker2.log_artifact(str(dummy_model_file), artifact_path="model")
        tracker2.log_metric("wape", 0.12)

    ver2 = register_model(
        mlflow_run_id=tracker2.mlflow_run_id,
        task="forecast",
        artifact_path="model",
        tracking_uri=temp_mlflow_uri,
    )
    assert ver2 == "2"

    # Promote Run 2 to champion
    promo2 = promote_champion(
        task="forecast",
        version=ver2,
        session=ml_db_session,
        tracking_uri=temp_mlflow_uri,
    )
    assert promo2["champion_version"] == "2"
    assert promo2["previous_champion_version"] == "1"

    # Check aliases
    client = MlflowClient(tracking_uri=temp_mlflow_uri)
    champ_mv = client.get_model_version_by_alias(REGISTERED_MODEL_FORECAST, ALIAS_CHAMPION)
    prev_mv = client.get_model_version_by_alias(REGISTERED_MODEL_FORECAST, ALIAS_PREVIOUS_CHAMPION)
    assert str(champ_mv.version) == "2"
    assert str(prev_mv.version) == "1"

    # Check DB status
    ml_db_session.expire_all()
    run1_db = ml_db_session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == tracker1.run_id)
    ).scalar_one()
    run2_db = ml_db_session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == tracker2.run_id)
    ).scalar_one()
    assert run1_db.is_champion is False
    assert run2_db.is_champion is True

    # Rollback champion (revert to previous champion)
    rollback_res = rollback_champion(
        task="forecast",
        session=ml_db_session,
        tracking_uri=temp_mlflow_uri,
    )
    assert rollback_res["champion_version"] == "1"

    ml_db_session.expire_all()
    run1_reverted = ml_db_session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == tracker1.run_id)
    ).scalar_one()
    run2_reverted = ml_db_session.execute(
        sa.select(ModelRun).where(ModelRun.run_id == tracker2.run_id)
    ).scalar_one()
    assert run1_reverted.is_champion is True
    assert run2_reverted.is_champion is False
