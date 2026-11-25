"""Pipeline trigger abstraction and execution adapters.

Provides a unified trigger interface for dispatching data ingestion, scoring,
training, and drift monitoring pipelines. Supports Noop (for testing/worker mode)
and Inline (for local synchronous execution) implementations.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request
from prefect.client.orchestration import PrefectClient

from retailpulse.config import get_settings
from retailpulse.db.models.ops import PipelineRun
from retailpulse.db.session import session_scope
from retailpulse.logging import get_logger

logger = get_logger(__name__)


class PipelineTrigger(ABC):
    """Abstract interface for pipeline execution triggers."""

    @abstractmethod
    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Trigger an asynchronous or background pipeline run.

        Args:
            flow_name: Name of the workflow ('etl', 'score_all', 'train_*', etc.).
            run_id: UUID of the ops.pipeline_run record.
            params: Optional execution parameters for the workflow.

        Returns:
            True if the trigger was accepted/dispatched successfully, False otherwise.
        """
        ...


class NoopPipelineTrigger(PipelineTrigger):
    """No-op trigger implementation for testing and worker-polled mode."""

    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Record the trigger call and return True without executing external flows."""
        del params
        logger.info(
            "NoopPipelineTrigger: trigger recorded",
            flow_name=flow_name,
            run_id=str(run_id),
        )
        return True


class InlinePipelineTrigger(PipelineTrigger):
    """Inline trigger implementation executing synchronous callbacks or recording runs."""

    def __init__(self, callback: Any = None) -> None:
        self.callback = callback
        self.dispatched_runs: list[tuple[str, uuid.UUID, dict[str, Any]]] = []

    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Record and optionally invoke an inline execution callback."""
        logger.info(
            "InlinePipelineTrigger: dispatching inline pipeline run",
            flow_name=flow_name,
            run_id=str(run_id),
        )
        p = params or {}
        self.dispatched_runs.append((flow_name, run_id, p))
        if self.callback is not None and callable(self.callback):
            res = self.callback(flow_name, run_id, p)
            if isinstance(res, bool):
                return res
        return True


class PrefectTrigger(PipelineTrigger):
    """Prefect orchestration deployment trigger for background execution."""

    def __init__(self, api_url: str | None = None) -> None:
        self.api_url = api_url

    async def trigger(
        self,
        flow_name: str,
        run_id: uuid.UUID,
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Dispatch the requested flow run to its served Prefect deployment.

        Args:
            flow_name: Target flow identifier ('etl', 'score_all', 'train_*', etc.).
            run_id: Unique pipeline run UUID.
            params: Optional parameter dict for the run.

        Returns:
            True if the run was successfully scheduled with Prefect.

        Raises:
            Exception: If Prefect server is unreachable or fails to schedule.
        """
        clean_flow = flow_name.strip().lower()
        deployment_name = f"{clean_flow}/{clean_flow}-deployment"
        endpoint = self.api_url or get_settings().prefect_api_url

        logger.info(
            "PrefectTrigger: dispatching flow to Prefect deployment",
            flow_name=clean_flow,
            deployment=deployment_name,
            run_id=str(run_id),
            api_url=endpoint,
        )

        async with PrefectClient(api=endpoint) as client:
            deployment = await client.read_deployment_by_name(deployment_name)
            flow_run = await client.create_flow_run_from_deployment(
                deployment_id=deployment.id,
                parameters={
                    "run_id": str(run_id),
                    "params": params or {},
                },
            )

            logger.info(
                "PrefectTrigger: flow run created successfully",
                flow_name=clean_flow,
                run_id=str(run_id),
                prefect_flow_run_id=str(flow_run.id),
                prefect_flow_run_name=flow_run.name,
            )

            try:
                with session_scope() as session:
                    run_rec = session.get(PipelineRun, run_id)
                    if run_rec is not None:
                        run_rec.prefect_flow_run_id = str(flow_run.id)
                        session.commit()
            except Exception as db_err:
                logger.warning(
                    "PrefectTrigger: failed to record prefect_flow_run_id in DB",
                    run_id=str(run_id),
                    error=str(db_err),
                )

            return True


def get_pipeline_trigger(request: Request) -> PipelineTrigger:
    """FastAPI dependency to retrieve the configured PipelineTrigger."""
    trigger = getattr(request.app.state, "pipeline_trigger", None)
    if trigger is None or not isinstance(trigger, PipelineTrigger):
        return NoopPipelineTrigger()
    return trigger
