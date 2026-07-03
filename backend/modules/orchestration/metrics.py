"""Runtime metrics collection for the DSS pipeline.

Collects pipeline duration, stage durations, module timings,
success/failure, and retry counts.  Future telemetry hooks.
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger("dss.orchestration.metrics")


@dataclass
class StageMetrics:
    """Timing and outcome data for a single pipeline stage."""

    stage_id: str = ""
    duration_ms: float = 0.0
    attempts: int = 1
    success: bool = False
    error: str | None = None


@dataclass
class RuntimeMetrics:
    """Aggregated metrics for one pipeline execution."""

    pipeline_id: str = ""
    request_id: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    stage_metrics: dict[str, StageMetrics] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def total_duration_ms(self) -> float:
        """Return the total pipeline duration in milliseconds."""
        if self.started_at == 0.0:
            return 0.0
        end = self.completed_at if self.completed_at > 0.0 else time.time()
        return (end - self.started_at) * 1000.0

    @property
    def success(self) -> bool:
        """Return True if the pipeline completed without errors."""
        return not self.errors and self.completed_at > 0.0


class MetricsCollector:
    """Collects runtime execution metrics for a single pipeline run."""

    def __init__(self) -> None:
        self._metrics: RuntimeMetrics = RuntimeMetrics()

    def start(
        self,
        pipeline_id: str = "",
        request_id: str = "",
    ) -> None:
        """Begin metrics collection for a new pipeline execution."""
        self._metrics = RuntimeMetrics(
            pipeline_id=pipeline_id,
            request_id=request_id,
            started_at=time.time(),
        )

    def complete(self) -> None:
        """Mark the pipeline execution as complete."""
        self._metrics.completed_at = time.time()

    def record_stage(
        self,
        stage_id: str,
        duration_ms: float,
        success: bool,
        error: str | None = None,
        attempts: int = 1,
    ) -> None:
        """Record metrics for a single pipeline stage.

        Parameters
        ----------
        stage_id:
            Identifier of the stage.
        duration_ms:
            Execution duration in milliseconds.
        success:
            Whether the stage completed successfully.
        error:
            Optional error message if the stage failed.
        attempts:
            Number of attempts made (default 1).
        """
        self._metrics.stage_metrics[stage_id] = StageMetrics(
            stage_id=stage_id,
            duration_ms=duration_ms,
            attempts=attempts,
            success=success,
            error=error,
        )

    def record_error(self, error: str) -> None:
        """Record a pipeline-level error."""
        self._metrics.errors.append(error)

    def record_warning(self, warning: str) -> None:
        """Record a pipeline-level warning."""
        self._metrics.warnings.append(warning)

    @property
    def metrics(self) -> RuntimeMetrics:
        """Return the accumulated metrics."""
        return self._metrics

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._metrics = RuntimeMetrics()
