"""Public entry point for the Orchestration Engine.

Coordinates the pipeline lifecycle while containing zero
business logic, AI, or module-specific code.
"""

import asyncio
import logging
from typing import Any

from backend.modules.orchestration.config import OrchestrationConfig, orch_config
from backend.modules.orchestration.context import PipelineContext
from backend.modules.orchestration.exceptions import (
    CancellationError,
    PipelineError,
)
from backend.modules.orchestration.interfaces import (
    PipelineInterface,
    RouterInterface,
    StageDefinition,
    WorkflowInterface,
)
from backend.modules.orchestration.pipeline import Pipeline
from backend.modules.orchestration.router import Router
from backend.modules.orchestration.state import PipelineState
from backend.modules.orchestration.workflow import Workflow

logger = logging.getLogger("dss.orchestration.service")


class OrchestrationService:
    """Central coordinator for the DSS intelligence pipeline.

    Accepts input data, creates an execution context, runs the
    configured pipeline, and returns the final context with results.

    Usage::

        service = OrchestrationService()
        service.router.register("computer_vision", my_vision_module)
        # ... register other modules ...

        context = await service.execute(image_metadata)
        print(context.get_stage_result("decision_engine"))
    """

    def __init__(
        self,
        pipeline: PipelineInterface | None = None,
        workflow: WorkflowInterface | None = None,
        router: RouterInterface | None = None,
        config: OrchestrationConfig | None = None,
    ) -> None:
        self._workflow = workflow or Workflow()
        self._pipeline = pipeline or Pipeline(self._workflow)
        self._router = router or Router()
        self._config = config or orch_config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def router(self) -> RouterInterface:
        """Access the module registry for registering modules."""
        return self._router

    @property
    def pipeline(self) -> PipelineInterface:
        """Access the pipeline for stage configuration."""
        return self._pipeline

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, input_data: Any) -> PipelineContext:
        """Run the full intelligence pipeline on *input_data*.

        Parameters
        ----------
        input_data:
            The initial input (e.g. an ``ImageMetadata`` contract).

        Returns
        -------
        PipelineContext
            The execution context containing all stage results and state.
        """
        context = PipelineContext(initial_input=input_data)
        context.state_machine.transition(PipelineState.RECEIVED)

        logger.info(
            "Pipeline %s starting (request %s)",
            context.pipeline_id,
            context.request_id,
        )

        try:
            context.state_machine.transition(PipelineState.RUNNING)

            await asyncio.wait_for(
                self._pipeline.execute(context),
                timeout=self._config.pipeline_timeout_seconds,
            )

            context.state_machine.transition(PipelineState.COMPLETED)
            logger.info("Pipeline %s completed", context.pipeline_id)

        except asyncio.TimeoutError:
            context.state_machine.transition(PipelineState.FAILED)
            msg = f"Pipeline timed out after {self._config.pipeline_timeout_seconds}s"
            context.errors.append(msg)
            logger.error("Pipeline %s: %s", context.pipeline_id, msg)

        except CancellationError:
            context.state_machine.transition(PipelineState.CANCELLED)
            logger.info("Pipeline %s was cancelled", context.pipeline_id)

        except PipelineError as exc:
            context.state_machine.transition(PipelineState.FAILED)
            context.errors.append(str(exc))
            logger.error("Pipeline %s failed: %s", context.pipeline_id, exc)

        return context

    async def cancel(self, context: PipelineContext) -> None:
        """Request cancellation of a running pipeline."""
        await self._workflow.cancel(context)

    # ------------------------------------------------------------------
    # Pipeline builder helpers
    # ------------------------------------------------------------------

    def add_stage(self, stage: StageDefinition) -> None:
        """Convenience: add a stage directly to the pipeline."""
        self._pipeline.add_stage(stage)

    def insert_stage(self, index: int, stage: StageDefinition) -> None:
        """Convenience: insert a stage at *index*."""
        self._pipeline.insert_stage(index, stage)
