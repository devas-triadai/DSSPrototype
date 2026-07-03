"""Shared execution context that flows through the entire pipeline."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from backend.modules.orchestration.state import PipelineState, StateMachine


@dataclass
class PipelineContext:
    """Request-scoped context for a single pipeline execution.

    Carries initial input, stores per-stage results, and tracks
    pipeline state, current stage, errors, and timing.
    """

    request_id: str = field(default_factory=lambda: str(uuid4()))
    pipeline_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    initial_input: Any = None

    state_machine: StateMachine = field(default_factory=StateMachine)
    _stage_results: dict[str, Any] = field(default_factory=dict)

    current_stage: str = ""
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    # ------------------------------------------------------------------
    # Stage result storage
    # ------------------------------------------------------------------

    def set_stage_result(self, key: str, value: Any) -> None:
        """Store the output of a pipeline stage.

        Parameters
        ----------
        key:
            Identifier for the stage (e.g. ``"computer_vision"``).
        value:
            Output produced by the stage (must be a contract model).
        """
        self._stage_results[key] = value

    def get_stage_result(self, key: str) -> Any:
        """Retrieve a previously stored stage result by *key*."""
        return self._stage_results.get(key)

    @property
    def last_result(self) -> Any:
        """Return the most recently stored stage result."""
        if not self._stage_results:
            return self.initial_input
        # Dicts preserve insertion order in Python 3.7+
        return list(self._stage_results.values())[-1]

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    @property
    def status(self) -> PipelineState:
        """Current pipeline state (delegated to the state machine)."""
        return self.state_machine.current

    def cancel(self) -> None:
        """Request cancellation at the next available opportunity."""
        self.cancelled = True
