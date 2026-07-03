"""Public runtime entry point for the DSS pipeline.

Responsible for executing the entire DSS pipeline from image
input to decision recommendation.  Coordinates module registry,
dependency resolution, pipeline execution, metrics collection,
and result assembly.  Contains zero AI.
"""

import logging
from typing import Any

from backend.contracts.models.detection import ImageMetadata
from backend.modules.orchestration.config import OrchestrationConfig, orch_config
from backend.modules.orchestration.dependency_resolver import DependencyResolver
from backend.modules.orchestration.health import HealthStatus, RuntimeHealth
from backend.modules.orchestration.metrics import MetricsCollector
from backend.modules.orchestration.module_registry import ModuleRegistry
from backend.modules.orchestration.pipeline_executor import PipelineExecutor
from backend.modules.orchestration.result_assembler import (
    ResultAssembler,
    RuntimeResult,
)

logger = logging.getLogger("dss.orchestration.runtime")


class Runtime:
    """Public runtime entry point for the DSS intelligence pipeline.

    Accepts an ``ImageMetadata`` input, executes all seven pipeline
    stages, and returns a ``RuntimeResult`` containing every contract
    model produced during execution.

    Usage::

        runtime = Runtime()
        runtime.register_module("computer_vision", vision_service)
        runtime.register_module("friendly", friendly_service)
        # ... register all modules ...

        result = await runtime.execute(image_metadata)
        print(result.decision)
    """

    def __init__(
        self,
        registry: ModuleRegistry | None = None,
        config: OrchestrationConfig | None = None,
    ) -> None:
        self._registry = registry or ModuleRegistry()
        self._config = config or orch_config
        self._resolver = DependencyResolver(self._registry)
        self._executor = PipelineExecutor(self._registry, self._config)
        self._assembler = ResultAssembler()
        self._health = RuntimeHealth(self._registry, self._resolver, self._config)

    # ------------------------------------------------------------------
    # Module registration
    # ------------------------------------------------------------------

    def register_module(self, module_type: str, module: Any) -> None:
        """Register a module with the runtime.

        Parameters
        ----------
        module_type:
            Unique key (e.g. ``"computer_vision"``, ``"friendly"``).
        module:
            The module instance implementing the corresponding interface.
        """
        self._registry.register(module_type, module)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, image: ImageMetadata) -> RuntimeResult:
        """Execute the full DSS pipeline on the given image.

        Parameters
        ----------
        image:
            Metadata describing the image to process.

        Returns
        -------
        RuntimeResult
            Complete pipeline output including all contract models
            and execution metadata.
        """
        metrics = MetricsCollector()
        metrics.start()

        # Verify dependencies before execution
        missing = self._resolver.verify()
        if missing:
            for mod in missing:
                metrics.record_error(
                    f"Cannot execute pipeline: module '{mod}' not registered"
                )
            metrics.complete()
            return ResultAssembler.assemble_error(
                errors=[f"Module not registered: '{mod}'" for mod in missing],
                metrics=metrics.metrics,
            )

        try:
            context = await self._executor.execute(image, metrics)
            metrics.complete()
            return self._assembler.assemble(context, metrics.metrics)

        except Exception as exc:
            metrics.record_error(str(exc))
            metrics.complete()
            return ResultAssembler.assemble_error(
                errors=[str(exc)],
                metrics=metrics.metrics,
            )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> HealthStatus:
        """Run runtime health checks.

        Returns
        -------
        HealthStatus
            Current health status of the runtime.
        """
        return self._health.check()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def registry(self) -> ModuleRegistry:
        """Access the underlying module registry."""
        return self._registry

    @property
    def is_ready(self) -> bool:
        """Return True if the runtime has all required modules registered."""
        return self._registry.is_complete()
