"""MLflow Model Registry integration, model promotion, and champion alias management.

Implements P2-09 & TRD §6:
- Registered models: retailpulse-forecast, retailpulse-segmentation, retailpulse-churn
- Aliases: champion and previous_champion
- Model registration from MLflow run artifacts
- Promotion: moves champion -> previous_champion, promotes new version,
  and updates ml.model_run.is_champion
- Rollback: restores previous champion version as champion
"""

from __future__ import annotations

import uuid
from typing import Any

import sqlalchemy as sa
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sqlalchemy.orm import Session

from retailpulse.config import get_settings
from retailpulse.db.models.ml import ModelRun
from retailpulse.logging import get_logger

logger = get_logger(__name__)

# Standard registered model names per TRD §6
REGISTERED_MODEL_FORECAST = "retailpulse-forecast"
REGISTERED_MODEL_SEGMENTATION = "retailpulse-segmentation"
REGISTERED_MODEL_CHURN = "retailpulse-churn"

TASK_TO_MODEL_NAME = {
    "forecast": REGISTERED_MODEL_FORECAST,
    "segmentation": REGISTERED_MODEL_SEGMENTATION,
    "churn": REGISTERED_MODEL_CHURN,
}

# Standard model aliases per TRD §6
ALIAS_CHAMPION = "champion"
ALIAS_PREVIOUS_CHAMPION = "previous_champion"


def _resolve_model_name(task: str | None, model_name: str | None) -> str:
    """Resolve registered model name from explicit name or task identifier."""
    if model_name:
        return model_name
    if task and task in TASK_TO_MODEL_NAME:
        return TASK_TO_MODEL_NAME[task]
    if task:
        return f"retailpulse-{task}"
    raise ValueError("Must provide either 'task' or 'model_name'")


def register_model(
    mlflow_run_id: str,
    task: str | None = None,
    *,
    model_name: str | None = None,
    artifact_path: str = "model",
    tags: dict[str, Any] | None = None,
    tracking_uri: str | None = None,
) -> str:
    """Register a new model version in MLflow Model Registry from a completed run.

    Args:
        mlflow_run_id: MLflow run ID containing the model artifact.
        task: ML task ('forecast', 'segmentation', 'churn').
        model_name: Optional explicit registered model name.
        artifact_path: Relative artifact path where weights were logged (defaults to 'model').
        tags: Optional tags to attach to the model version.
        tracking_uri: Optional MLflow tracking URI.

    Returns:
        Version number string of the newly registered model version.
    """
    effective_uri = tracking_uri or get_settings().mlflow_tracking_uri
    client = MlflowClient(tracking_uri=effective_uri)
    reg_name = _resolve_model_name(task, model_name)

    # Ensure registered model exists
    try:
        client.get_registered_model(reg_name)
    except MlflowException:
        logger.info("creating new registered model", model_name=reg_name)
        client.create_registered_model(reg_name)

    source_uri = f"runs:/{mlflow_run_id}/{artifact_path}"
    string_tags = {k: str(v) for k, v in tags.items()} if tags else None

    mv = client.create_model_version(
        name=reg_name,
        source=source_uri,
        run_id=mlflow_run_id,
        tags=string_tags,
    )
    logger.info(
        "registered model version",
        model_name=reg_name,
        version=mv.version,
        run_id=mlflow_run_id,
    )
    return str(mv.version)


def promote_champion(
    task: str,
    version: str,
    *,
    session: Session | None = None,
    model_name: str | None = None,
    tracking_uri: str | None = None,
    actor_id: uuid.UUID | None = None,
    is_auto: bool = True,
) -> dict[str, Any]:
    """Promote a model version to 'champion' alias and update database mirrors.

    Per TRD §6:
    - Moves current 'champion' alias to 'previous_champion'.
    - Assigns 'champion' alias to the specified version.
    - Updates ml.model_run.is_champion in database transaction.

    Args:
        task: Target ML task ('forecast', 'segmentation', 'churn').
        version: Model version number string to promote.
        session: Active database session for updating ml.model_run.
        model_name: Optional explicit model name.
        tracking_uri: Optional MLflow tracking URI.
        actor_id: Optional user UUID initiating the promotion.

    Returns:
        Summary dictionary with promotion details.
    """
    effective_uri = tracking_uri or get_settings().mlflow_tracking_uri
    client = MlflowClient(tracking_uri=effective_uri)
    reg_name = _resolve_model_name(task, model_name)

    # 1. Check current champion version
    previous_version: str | None = None
    try:
        current_champ = client.get_model_version_by_alias(reg_name, ALIAS_CHAMPION)
        previous_version = str(current_champ.version)
    except MlflowException:
        pass

    # 2. Update MLflow aliases
    if previous_version is not None and previous_version != version:
        client.set_registered_model_alias(reg_name, ALIAS_PREVIOUS_CHAMPION, previous_version)

    client.set_registered_model_alias(reg_name, ALIAS_CHAMPION, version)

    # 3. Mirror status to database if session is provided
    target_run_id: uuid.UUID | None = None
    if session is not None:
        # Clear previous champion in DB
        session.execute(
            sa.update(ModelRun)
            .where(ModelRun.task == task, ModelRun.is_champion.is_(True))
            .values(is_champion=False)
        )

        # Look up run ID corresponding to promoted version
        promoted_mv = client.get_model_version(reg_name, version)
        matching_run = session.execute(
            sa.select(ModelRun).where(ModelRun.mlflow_run_id == promoted_mv.run_id)
        ).scalar_one_or_none()

        if matching_run is not None:
            matching_run.is_champion = True
            matching_run.mlflow_model_name = reg_name
            matching_run.mlflow_model_version = version
            target_run_id = matching_run.run_id

            try:
                from retailpulse.db.models.ops import AuditLog  # noqa: PLC0415

                action_name = "promote (auto)" if is_auto else "promote"
                audit_entry = AuditLog(
                    action=action_name,
                    user_id=actor_id,
                    resource=f"model:{task}:{target_run_id}",
                    details={
                        "task": task,
                        "model_name": reg_name,
                        "champion_version": version,
                        "previous_champion_version": previous_version,
                        "is_auto": is_auto,
                    },
                )
                session.add(audit_entry)
                session.flush()
            except Exception as audit_err:
                logger.warning("Failed to record promotion audit log", error=str(audit_err))
                session.rollback()

        session.commit()

    logger.info(
        "promoted champion model",
        task=task,
        model_name=reg_name,
        champion_version=version,
        previous_champion_version=previous_version,
    )

    return {
        "task": task,
        "model_name": reg_name,
        "champion_version": version,
        "previous_champion_version": previous_version,
        "model_run_id": str(target_run_id) if target_run_id else None,
        "actor_id": str(actor_id) if actor_id else None,
    }


def rollback_champion(
    task: str,
    *,
    session: Session | None = None,
    model_name: str | None = None,
    tracking_uri: str | None = None,
    actor_id: uuid.UUID | None = None,
) -> dict[str, Any]:
    """Roll back champion model to the previous champion version.

    Per TRD §6:
    - Finds the version tagged with alias 'previous_champion'.
    - Calls promote_champion on that version.
    """
    effective_uri = tracking_uri or get_settings().mlflow_tracking_uri
    client = MlflowClient(tracking_uri=effective_uri)
    reg_name = _resolve_model_name(task, model_name)

    try:
        prev_champ = client.get_model_version_by_alias(reg_name, ALIAS_PREVIOUS_CHAMPION)
    except MlflowException as exc:
        raise ValueError(
            f"No previous champion found for model '{reg_name}' to rollback to"
        ) from exc

    prev_version = str(prev_champ.version)
    return promote_champion(
        task=task,
        version=prev_version,
        session=session,
        model_name=reg_name,
        tracking_uri=effective_uri,
        actor_id=actor_id,
    )


def get_champion_version(
    task: str,
    *,
    model_name: str | None = None,
    tracking_uri: str | None = None,
) -> str | None:
    """Retrieve version string of current champion from MLflow."""
    effective_uri = tracking_uri or get_settings().mlflow_tracking_uri
    client = MlflowClient(tracking_uri=effective_uri)
    reg_name = _resolve_model_name(task, model_name)
    try:
        mv = client.get_model_version_by_alias(reg_name, ALIAS_CHAMPION)
        return str(mv.version)
    except MlflowException:
        return None


def get_champion_run(task: str, session: Session) -> ModelRun | None:
    """Retrieve active champion ModelRun record from the database."""
    stmt = sa.select(ModelRun).where(ModelRun.task == task, ModelRun.is_champion.is_(True))
    return session.execute(stmt).scalar_one_or_none()
