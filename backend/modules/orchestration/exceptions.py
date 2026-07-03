"""Orchestration-specific exceptions."""


class PipelineError(RuntimeError):
    """Base exception for all pipeline errors."""


class WorkflowError(PipelineError):
    """Raised when a workflow operation fails after exhausting retries."""


class RoutingError(PipelineError):
    """Raised when a required module is not registered with the router."""


class TimeoutError(PipelineError):
    """Raised when a stage or pipeline exceeds its allowed duration."""


class CancellationError(PipelineError):
    """Raised when the pipeline is explicitly cancelled."""


class StateTransitionError(PipelineError):
    """Raised when an invalid pipeline-state transition is attempted."""
