"""Pipeline executor for the Runtime Integration Layer.

Coordinates execution of the full DSS pipeline in order:

    Computer Vision
    → Friendly Knowledge
    → Enemy Knowledge
    → Terrain Knowledge
    → Fusion Engine
    → Threat Assessment
    → Decision Engine

Supports future parallel execution where dependencies allow.
"""

import logging
import time
from typing import Any

from backend.contracts.models.detection import ImageMetadata
from backend.modules.orchestration.config import orch_config
from backend.modules.orchestration.context import PipelineContext
from backend.modules.orchestration.interfaces import StageDefinition
from backend.modules.orchestration.metrics import MetricsCollector
from backend.modules.orchestration.module_registry import ModuleRegistry
from backend.modules.orchestration.state import PipelineState
from backend.modules.orchestration.workflow import Workflow

logger = logging.getLogger("dss.orchestration.pipeline_executor")

_STAGE_IDS = (
    "detection",
    "friendly",
    "enemy",
    "terrain",
    "fusion",
    "threat",
    "decision",
)


class PipelineExecutor:
    """Coordinates execution of the full DSS pipeline.

    Builds reusable stage callables that resolve module instances
    from the registry at call time.  Each stage is executed through
    the ``Workflow`` layer for configurable retry and timeout.

    Future parallel execution: stages *friendly*, *enemy*, and
    *terrain* share no mutual dependencies and can be scheduled
    concurrently via ``asyncio.gather``.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        config: Any | None = None,
    ) -> None:
        self._registry = registry
        self._config = config or orch_config
        self._workflow = Workflow()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        input_data: ImageMetadata,
        metrics: MetricsCollector | None = None,
    ) -> PipelineContext:
        """Execute the full pipeline and return the result context.

        Parameters
        ----------
        input_data:
            The image metadata to process through the pipeline.
        metrics:
            Optional collector for per-stage timing metrics.

        Returns
        -------
        PipelineContext
            Execution context populated with all stage results.
        """
        context = PipelineContext(initial_input=input_data)
        context.state_machine.transition(PipelineState.RECEIVED)
        context.state_machine.transition(PipelineState.RUNNING)

        try:
            detection = await self._run_stage(
                stage_id=_STAGE_IDS[0],
                stage_name="Computer Vision",
                input_data=input_data,
                execute_fn=lambda _: self._registry.get_vision().process_image(
                    input_data
                ),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[0], detection)

            friendly = await self._run_stage(
                stage_id=_STAGE_IDS[1],
                stage_name="Friendly Knowledge",
                input_data=detection,
                execute_fn=lambda _: self._registry.get_friendly().analyze_friendly(
                    detection
                ),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[1], friendly)

            enemy = await self._run_stage(
                stage_id=_STAGE_IDS[2],
                stage_name="Enemy Knowledge",
                input_data=detection,
                execute_fn=lambda _: self._registry.get_enemy().analyze_enemy(
                    detection
                ),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[2], enemy)

            terrain = await self._run_stage(
                stage_id=_STAGE_IDS[3],
                stage_name="Terrain Knowledge",
                input_data=detection,
                execute_fn=lambda _: self._registry.get_terrain().analyze_terrain(
                    detection
                ),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[3], terrain)

            fusion = await self._run_stage(
                stage_id=_STAGE_IDS[4],
                stage_name="Fusion Engine",
                input_data=None,
                execute_fn=lambda _: self._registry.get_fusion().fuse_intelligence(
                    friendly, enemy, terrain
                ),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[4], fusion)

            threat = await self._run_stage(
                stage_id=_STAGE_IDS[5],
                stage_name="Threat Assessment",
                input_data=fusion,
                execute_fn=lambda _: self._registry.get_fusion().assess_threat(
                    fusion
                ),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[5], threat)

            decision = await self._run_stage(
                stage_id=_STAGE_IDS[6],
                stage_name="Decision Engine",
                input_data=None,
                execute_fn=lambda _: self._registry.get_decision(
                ).generate_recommendations(threat, fusion),
                context=context,
                metrics=metrics,
            )
            context.set_stage_result(_STAGE_IDS[6], decision)

            context.state_machine.transition(PipelineState.COMPLETED)

        except Exception as exc:
            context.state_machine.transition(PipelineState.FAILED)
            context.errors.append(str(exc))
            logger.error("Pipeline execution failed: %s", exc)

        return context

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    async def _run_stage(
        self,
        stage_id: str,
        stage_name: str,
        input_data: Any,
        execute_fn: Any,
        context: PipelineContext,
        metrics: MetricsCollector | None,
    ) -> Any:
        """Execute a single pipeline stage with retry and timeout."""
        stage = StageDefinition(
            stage_id=stage_id,
            name=stage_name,
            input_key="",
            output_key=stage_id,
            execute=execute_fn,
            timeout_seconds=self._config.default_stage_timeout_seconds,
            retry_count=self._config.default_retry_count,
            retry_delay_seconds=self._config.default_retry_delay_seconds,
        )

        context.current_stage = stage_id
        start = time.monotonic()

        try:
            result = await self._workflow.execute_stage(stage, input_data)

            if metrics:
                elapsed = (time.monotonic() - start) * 1000.0
                metrics.record_stage(stage_id, elapsed, success=True)

            logger.debug("Stage '%s' completed successfully", stage_id)
            return result

        except Exception as exc:
            if metrics:
                elapsed = (time.monotonic() - start) * 1000.0
                metrics.record_stage(stage_id, elapsed, success=False, error=str(exc))

            logger.error("Stage '%s' failed: %s", stage_id, exc)
            raise

    # ------------------------------------------------------------------
    # Future parallel execution support
    # ------------------------------------------------------------------

    @staticmethod
    def parallel_stage_groups() -> list[list[str]]:
        """Return stage groups that can be executed in parallel.

        Returns
        -------
        list[list[str]]
            Groups of stage IDs that share no mutual dependencies
            and can safely run concurrently.
        """
        return [
            ["computer_vision"],
            ["friendly", "enemy", "terrain"],
            ["fusion"],
            ["decision"],
        ]
