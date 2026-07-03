"""Configurable execution pipeline that sequences stages and routes data.

New stages can be added at any position without rewriting the pipeline.
"""

from backend.modules.orchestration.context import PipelineContext
from backend.modules.orchestration.exceptions import CancellationError, PipelineError
from backend.modules.orchestration.interfaces import (
    PipelineInterface,
    StageDefinition,
    WorkflowInterface,
)


class Pipeline(PipelineInterface):
    """Ordered sequence of stages executed one after another.

    Each stage reads its input from the shared ``PipelineContext``,
    executes, and writes its output back.  Stages are decoupled from
    each other — they communicate only through the context.
    """

    def __init__(self, workflow: WorkflowInterface) -> None:
        self._workflow = workflow
        self._stages: list[StageDefinition] = []

    # ------------------------------------------------------------------
    # Stage management
    # ------------------------------------------------------------------

    def add_stage(self, stage: StageDefinition) -> None:
        """Append *stage* to the end of the pipeline."""
        self._stages.append(stage)

    def insert_stage(self, index: int, stage: StageDefinition) -> None:
        """Insert *stage* at *index* (0‑based)."""
        self._stages.insert(index, stage)

    def remove_stage(self, stage_id: str) -> None:
        """Remove the stage identified by *stage_id*."""
        self._stages = [s for s in self._stages if s.stage_id != stage_id]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, context: PipelineContext) -> None:
        """Iterate through every stage in order.

        For each stage:
          1. Check cancellation.
          2. Read input from context (initial input for the first stage).
          3. Execute via the workflow layer (retries, timeout).
          4. Store output in context.

        Parameters
        ----------
        context:
            Shared execution context carrying input and state.
        """
        for stage in self._stages:
            if context.cancelled:
                raise CancellationError(
                    f"Pipeline cancelled before stage '{stage.stage_id}'"
                )

            context.current_stage = stage.stage_id

            # Resolve input — first stage uses initial_input,
            # subsequent stages read the previous stage's output.
            input_data: object = (
                context.initial_input
                if not stage.input_key
                else context.get_stage_result(stage.input_key)
            )
            if input_data is None:
                raise PipelineError(
                    f"Stage '{stage.stage_id}': no input found for key '{stage.input_key}'"
                )

            output = await self._workflow.execute_stage(stage, input_data)
            context.set_stage_result(stage.output_key, output)

    @property
    def stages(self) -> list[StageDefinition]:
        """Return a copy of the current stage list."""
        return list(self._stages)
