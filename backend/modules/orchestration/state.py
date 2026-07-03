"""Pipeline state machine with enforceable transition rules."""

from enum import Enum

from backend.modules.orchestration.exceptions import StateTransitionError


class PipelineState(str, Enum):
    """All possible states of a pipeline execution."""

    PENDING = "pending"
    RECEIVED = "received"
    RUNNING = "running"
    WAITING = "waiting"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# Allowable transitions: {from_state: {to_state, ...}}
_TRANSITIONS: dict[PipelineState, set[PipelineState]] = {
    PipelineState.PENDING: {PipelineState.RECEIVED},
    PipelineState.RECEIVED: {PipelineState.RUNNING},
    PipelineState.RUNNING: {
        PipelineState.WAITING,
        PipelineState.FAILED,
        PipelineState.COMPLETED,
        PipelineState.CANCELLED,
    },
    PipelineState.WAITING: {PipelineState.RUNNING, PipelineState.CANCELLED},
    PipelineState.FAILED: set(),
    PipelineState.COMPLETED: set(),
    PipelineState.CANCELLED: set(),
}


class StateMachine:
    """Tracks and validates pipeline-state transitions.

    Usage::

        sm = StateMachine()
        sm.transition(PipelineState.RECEIVED)
        sm.transition(PipelineState.RUNNING)
    """

    def __init__(self) -> None:
        self._state: PipelineState = PipelineState.PENDING

    def transition(self, new_state: PipelineState) -> None:
        """Move to *new_state* if the transition is allowed.

        Raises
        ------
        StateTransitionError
            If the transition is not in the allowed set.
        """
        allowed = _TRANSITIONS[self._state]
        if new_state not in allowed:
            raise StateTransitionError(
                f"Cannot transition from {self._state.value} to {new_state.value}"
            )
        self._state = new_state

    @property
    def current(self) -> PipelineState:
        """Return the current state."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if the current state is terminal (no transitions out)."""
        return len(_TRANSITIONS[self._state]) == 0
