"""Stage-level execution with retry, timeout, and cancellation support."""

import asyncio
import logging
from typing import Any

from backend.modules.orchestration.config import orch_config
from backend.modules.orchestration.context import PipelineContext
from backend.modules.orchestration.exceptions import (
    CancellationError,
    TimeoutError,
    WorkflowError,
)
from backend.modules.orchestration.interfaces import StageDefinition, WorkflowInterface

logger = logging.getLogger("dss.orchestration.workflow")


class Workflow(WorkflowInterface):
    """Executes a single pipeline stage with configurable retries and timeout.

    The workflow layer is responsible for:
      - Retrying failed stages up to a configurable limit.
      - Enforcing per-stage timeouts.
      - Respecting cancellation requests between retry attempts.
      - Wrapping raw exceptions into typed ``WorkflowException``.
    """

    def __init__(self) -> None:
        self._config = orch_config

    async def execute_stage(
        self,
        stage: StageDefinition,
        input_data: Any,
    ) -> Any:
        """Run *stage* with retry and timeout.

        Parameters
        ----------
        stage:
            The stage descriptor including its execute callable.
        input_data:
            Input to forward to *stage.execute*.

        Returns
        -------
        Any
            The output produced by the stage.

        Raises
        ------
        TimeoutError
            If all retry attempts time out.
        WorkflowError
            If all retry attempts fail with non‑timeout errors.
        CancellationError
            If the pipeline is cancelled between retries.
        """
        timeout = stage.timeout_seconds or self._config.default_stage_timeout_seconds
        retries = stage.retry_count if stage.retry_count >= 0 else self._config.default_retry_count
        delay = stage.retry_delay_seconds or self._config.default_retry_delay_seconds

        last_exc: Exception | None = None

        for attempt in range(retries + 1):
            logger.debug(
                "Stage '%s' attempt %d/%d",
                stage.stage_id,
                attempt + 1,
                retries + 1,
            )
            try:
                result = await asyncio.wait_for(
                    stage.execute(input_data),
                    timeout=timeout,
                )
                return result

            except asyncio.TimeoutError:
                last_exc = TimeoutError(
                    f"Stage '{stage.stage_id}' timed out after {timeout}s "
                    f"(attempt {attempt + 1}/{retries + 1})"
                )
                logger.warning(str(last_exc))

            except asyncio.CancelledError:
                raise CancellationError(
                    f"Stage '{stage.stage_id}' was cancelled"
                ) from None

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Stage '%s' failed (attempt %d/%d): %s",
                    stage.stage_id,
                    attempt + 1,
                    retries + 1,
                    exc,
                )

            # Wait before retrying unless this was the final attempt
            if attempt < retries:
                await asyncio.sleep(delay)

        # All attempts exhausted
        if isinstance(last_exc, TimeoutError):
            raise last_exc
        raise WorkflowError(
            f"Stage '{stage.stage_id}' failed after {retries + 1} attempt(s)"
        ) from last_exc

    async def cancel(self, context: PipelineContext) -> None:
        """Mark the pipeline as cancelled.

        The cancellation takes effect at the next stage boundary
        when the pipeline checks ``context.cancelled``.
        """
        logger.info("Cancelling pipeline %s", context.pipeline_id)
        context.cancel()
