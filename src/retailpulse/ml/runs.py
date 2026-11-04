"""MLflow run management, parameter/metric tracking, and database mirroring.

Implements P2-09 & TRD §6:
- Experiment names: retailpulse/forecast, retailpulse/segmentation, retailpulse/churn
- Standard tags: git_sha, dataset_hash, data_as_of, seed, library_versions
- Parameter, metric, dictionary, and artifact logging via MLflow client
- Mirroring to ml.model_run and ml.forecast_metric in database
- Context manager track_model_run with automatic lifecycle management
"""

from __future__ import annotations

import datetime
import importlib.metadata
import os
import shutil
import subprocess
import uuid
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import RunStatus
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.ml import ForecastMetric, ModelRun
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Standard experiment names per TRD §6
EXPERIMENT_FORECAST = "retailpulse/forecast"
EXPERIMENT_SEGMENTATION = "retailpulse/segmentation"
EXPERIMENT_CHURN = "retailpulse/churn"

TASK_TO_EXPERIMENT = {
    "forecast": EXPERIMENT_FORECAST,
    "segmentation": EXPERIMENT_SEGMENTATION,
    "churn": EXPERIMENT_CHURN,
}


def get_git_sha() -> str:
    """Retrieve current Git commit SHA from environment or subprocess."""
    env_sha = os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha.strip()
    git_bin = shutil.which("git") or "git"
    try:
        res = subprocess.run(  # noqa: S603
            [git_bin, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_library_versions() -> str:
    """Capture runtime versions of core ML dependencies for reproducibility tags."""
    versions: list[str] = [f"mlflow={mlflow.__version__}"]
    for pkg in ("lightgbm", "prophet", "torch"):
        try:
            ver = importlib.metadata.version(pkg)
            versions.append(f"{pkg}={ver}")
        except importlib.metadata.PackageNotFoundError:
            pass

    return ",".join(versions)


class ModelRunTracker:
    """Wrapper managing an active MLflow run and its corresponding database mirrors."""

    def __init__(
        self,
        *,
        run_id: uuid.UUID,
        mlflow_run_id: str,
        task: str,
        algorithm: str,
        experiment_name: str,
        tracking_uri: str,
        session: Session | None = None,
    ) -> None:
        self.run_id = run_id
        self.mlflow_run_id = mlflow_run_id
        self.task = task
        self.algorithm = algorithm
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.session = session

        self._client = MlflowClient(tracking_uri=tracking_uri)
        self._logged_params: dict[str, Any] = {}
        self._logged_metrics: dict[str, float] = {}
        self._is_finished = False

    def log_param(self, key: str, value: Any) -> None:
        """Log a parameter to MLflow and record locally for DB mirroring."""
        str_val = str(value)
        self._client.log_param(self.mlflow_run_id, key, str_val)
        self._logged_params[key] = value

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple parameters at once."""
        for k, v in params.items():
            self.log_param(k, v)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a numerical metric to MLflow and record locally for DB mirroring."""
        float_val = float(value)
        self._client.log_metric(self.mlflow_run_id, key, float_val, step=step)
        self._logged_metrics[key] = float_val

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step=step)

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the active MLflow run."""
        self._client.set_tag(self.mlflow_run_id, key, str(value))

    def set_tags(self, tags: dict[str, Any]) -> None:
        """Set multiple tags on the active MLflow run."""
        for k, v in tags.items():
            self.set_tag(k, v)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        """Log an artifact file or directory to MLflow."""
        self._client.log_artifact(self.mlflow_run_id, str(local_path), artifact_path=artifact_path)

    def log_dict(self, dictionary: dict[str, Any], artifact_file: str) -> None:
        """Serialize a dictionary to JSON and log as an MLflow artifact."""
        self._client.log_dict(self.mlflow_run_id, dictionary, artifact_file)

    def mirror_forecast_metrics(
        self,
        metrics: pd.DataFrame | Sequence[dict[str, Any]],
        session: Session | None = None,
    ) -> int:
        """Mirror scoped forecast metrics into the database table ml.forecast_metric.

        Args:
            metrics: DataFrame or records with: scope, scope_key, metric, value, n_series.
            session: Optional DB session (defaults to tracker's session).

        Returns:
            Number of metric records inserted.
        """
        active_sess = session or self.session
        if active_sess is None:
            logger.warning("no database session available to mirror forecast metrics")
            return 0

        if isinstance(metrics, pd.DataFrame):
            records = metrics.to_dict(orient="records")
        else:
            records = list(metrics)

        count = 0
        for rec in records:
            obj = ForecastMetric(
                model_run_id=self.run_id,
                scope=str(rec["scope"]),
                scope_key=str(rec["scope_key"]),
                metric=str(rec["metric"]),
                value=float(rec["value"]),
                n_series=int(rec["n_series"]) if rec.get("n_series") is not None else None,
            )
            active_sess.merge(obj)
            count += 1

        active_sess.commit()
        logger.info(
            "mirrored forecast metrics to database",
            model_run_id=str(self.run_id),
            count=count,
        )
        return count

    def finish(self, status: str = "succeeded", error: str | None = None) -> None:
        """Finalize MLflow run status and mirror final parameters and metrics into ml.model_run."""
        if self._is_finished:
            return

        mlflow_status = RunStatus.FINISHED if status == "succeeded" else RunStatus.FAILED
        self._client.set_terminated(self.mlflow_run_id, status=RunStatus.to_string(mlflow_status))

        if self.session is not None:
            db_run = self.session.get(ModelRun, self.run_id)
            if db_run is not None:
                db_run.status = status
                db_run.params = dict(self._logged_params)
                db_metrics: dict[str, Any] = dict(self._logged_metrics)
                if error:
                    db_metrics["error"] = error
                db_run.metrics = db_metrics
                self.session.commit()

        self._is_finished = True
        logger.info(
            "finished model run tracking",
            model_run_id=str(self.run_id),
            mlflow_run_id=self.mlflow_run_id,
            status=status,
        )


@contextmanager
def track_model_run(
    task: str,
    algorithm: str,
    *,
    data_as_of: datetime.date | None = None,
    dataset_hash: str | None = None,
    seed: int | None = None,
    data_start: datetime.date | None = None,
    data_end: datetime.date | None = None,
    session: Session | None = None,
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
    run_name: str | None = None,
    git_sha: str | None = None,
    trained_by: uuid.UUID | None = None,
    tags: dict[str, Any] | None = None,
) -> Generator[ModelRunTracker, None, None]:
    """Context manager for tracking an ML training run in MLflow and mirroring to PostgreSQL.

    Per TRD §6:
    - Experiments: retailpulse/forecast, retailpulse/segmentation, retailpulse/churn
    - Default tags: git_sha, dataset_hash, data_as_of, seed, library_versions
    - Mirrors lifecycle and metadata to ml.model_run.
    """
    effective_uri = tracking_uri or get_settings().mlflow_tracking_uri
    client = MlflowClient(tracking_uri=effective_uri)

    exp_name = experiment_name or TASK_TO_EXPERIMENT.get(task, f"retailpulse/{task}")
    exp = client.get_experiment_by_name(exp_name)
    exp_id = client.create_experiment(exp_name) if exp is None else exp.experiment_id

    # Create run tags
    run_tags: dict[str, str] = {
        "task": task,
        "algorithm": algorithm,
        "git_sha": git_sha or get_git_sha(),
        "library_versions": get_library_versions(),
    }
    if dataset_hash:
        run_tags["dataset_hash"] = dataset_hash
    if data_as_of:
        run_tags["data_as_of"] = str(data_as_of)
    if seed is not None:
        run_tags["seed"] = str(seed)
    if tags:
        run_tags.update({k: str(v) for k, v in tags.items()})

    mlflow_run = client.create_run(
        experiment_id=exp_id,
        run_name=run_name or f"{algorithm}_{data_as_of or 'run'}",
        tags=run_tags,
    )

    app_run_id = uuid.uuid4()

    # Mirror initial row in ml.model_run if session provided
    if session is not None:
        db_model_run = ModelRun(
            run_id=app_run_id,
            task=task,
            algorithm=algorithm,
            mlflow_run_id=mlflow_run.info.run_id,
            status="running",
            is_champion=False,
            data_start=data_start,
            data_end=data_end,
            params={},
            metrics={},
            git_sha=run_tags.get("git_sha"),
            dataset_hash=dataset_hash,
            seed=seed,
            trained_by=trained_by,
        )
        session.add(db_model_run)
        session.commit()

    tracker = ModelRunTracker(
        run_id=app_run_id,
        mlflow_run_id=mlflow_run.info.run_id,
        task=task,
        algorithm=algorithm,
        experiment_name=exp_name,
        tracking_uri=effective_uri,
        session=session,
    )

    try:
        yield tracker
        tracker.finish(status="succeeded")
    except Exception as exc:
        tracker.finish(status="failed", error=str(exc))
        raise
