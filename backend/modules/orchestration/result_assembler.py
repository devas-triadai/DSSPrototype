"""Result assembly for the Runtime Integration Layer.

Collects all pipeline stage outputs into a single runtime result
object with execution metadata.  Uses runtime-only result models
— no modifications to ``backend.contracts``.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.contracts.models.analysis import EnemyAnalysis, FriendlyAnalysis, TerrainAnalysis
from backend.contracts.models.decision import DecisionRecommendation
from backend.contracts.models.detection import DetectionResult
from backend.contracts.models.fusion import FusionResult, ThreatAssessment
from backend.modules.orchestration.context import PipelineContext
from backend.modules.orchestration.metrics import RuntimeMetrics

logger = logging.getLogger("dss.orchestration.result_assembler")


@dataclass(frozen=True)
class ExecutionMetadata:
    """Execution metadata for a single pipeline run.

    Contains timing, status, and diagnostic information
    collected during execution.  Runtime-only type.
    """

    pipeline_id: str = ""
    request_id: str = ""
    status: str = "unknown"
    total_duration_ms: float = 0.0
    stage_durations: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuntimeResult:
    """Complete output of a single DSS pipeline execution.

    Contains every contract model produced during execution along
    with execution metadata.  All result fields are Optional to
    support partial pipeline failures.  Runtime-only type.
    """

    detection: DetectionResult | None = None
    friendly: FriendlyAnalysis | None = None
    enemy: EnemyAnalysis | None = None
    terrain: TerrainAnalysis | None = None
    fusion: FusionResult | None = None
    threat: ThreatAssessment | None = None
    decision: DecisionRecommendation | None = None
    metadata: ExecutionMetadata = field(
        default_factory=lambda: ExecutionMetadata()
    )

    @property
    def has_decision(self) -> bool:
        """Return True if a decision recommendation was produced."""
        return self.decision is not None

    @property
    def success(self) -> bool:
        """Return True if the pipeline completed without errors."""
        return self.metadata.status == "completed"


class ResultAssembler:
    """Assembles pipeline stage results into a ``RuntimeResult``.

    Extracts each stage output from the ``PipelineContext`` and
    combines them with timing metadata from the metrics collector.
    """

    def assemble(
        self,
        context: PipelineContext,
        metrics: RuntimeMetrics | None = None,
    ) -> RuntimeResult:
        """Collect all stage results into a single ``RuntimeResult``.

        Parameters
        ----------
        context:
            The pipeline execution context with stage results.
        metrics:
            Optional runtime metrics for timing data.

        Returns
        -------
        RuntimeResult
            Complete result with all contract models and metadata.
        """
        metadata = self._build_metadata(context, metrics)

        return RuntimeResult(
            detection=self._safe_get(context, "detection"),
            friendly=self._safe_get(context, "friendly"),
            enemy=self._safe_get(context, "enemy"),
            terrain=self._safe_get(context, "terrain"),
            fusion=self._safe_get(context, "fusion"),
            threat=self._safe_get(context, "threat"),
            decision=self._safe_get(context, "decision"),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        context: PipelineContext,
        metrics: RuntimeMetrics | None,
    ) -> ExecutionMetadata:
        """Build execution metadata from context and metrics."""
        stage_durations: dict[str, float] = {}
        if metrics and metrics.stage_metrics:
            stage_durations = {
                sid: sm.duration_ms
                for sid, sm in metrics.stage_metrics.items()
            }

        errors = (
            list(context.errors)
            if context.errors
            else (list(metrics.errors) if metrics else [])
        )
        warnings = list(metrics.warnings) if metrics else []

        return ExecutionMetadata(
            pipeline_id=context.pipeline_id,
            request_id=context.request_id,
            status=context.status.value if context.status else "unknown",
            total_duration_ms=metrics.total_duration_ms if metrics else 0.0,
            stage_durations=stage_durations,
            errors=errors,
            warnings=warnings,
        )

    @staticmethod
    def assemble_error(
        errors: list[str],
        metrics: RuntimeMetrics | None = None,
        pipeline_id: str = "",
        request_id: str = "",
    ) -> RuntimeResult:
        """Build a ``RuntimeResult`` representing a failed execution.

        Parameters
        ----------
        errors:
            List of error messages describing the failure.
        metrics:
            Optional runtime metrics.
        pipeline_id:
            Optional pipeline identifier.
        request_id:
            Optional request identifier.

        Returns
        -------
        RuntimeResult
            Result with status ``"failed"`` and populated error list.
        """
        metadata = ExecutionMetadata(
            pipeline_id=pipeline_id,
            request_id=request_id,
            status="failed",
            total_duration_ms=metrics.total_duration_ms if metrics else 0.0,
            errors=errors,
            warnings=list(metrics.warnings) if metrics else [],
        )
        return RuntimeResult(metadata=metadata)

    @staticmethod
    def _safe_get(context: PipelineContext, key: str) -> Any:
        """Retrieve a stage result from context without raising."""
        try:
            return context.get_stage_result(key)
        except Exception:
            return None
