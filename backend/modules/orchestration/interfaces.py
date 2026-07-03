"""Abstract interfaces for every orchestration component.

Concrete implementations depend on these contracts exclusively.
"""

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from backend.modules.orchestration.context import PipelineContext

# ---------------------------------------------------------------------------
# Stage definition
# ---------------------------------------------------------------------------


@dataclass
class StageDefinition:
    """Descriptor for a single stage in the execution pipeline.

    Each stage reads its input from the context (keyed by *input_key*),
    invokes the callable, and stores the output under *output_key*.
    """

    stage_id: str
    name: str
    input_key: str
    output_key: str
    execute: Callable[[Any], Awaitable[Any]] = field(compare=False)
    timeout_seconds: float = 60.0
    retry_count: int = 2
    retry_delay_seconds: float = 1.0

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise ValueError("stage_id must not be empty")


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class PipelineInterface(ABC):
    """Contract for the execution pipeline."""

    @abstractmethod
    def add_stage(self, stage: StageDefinition) -> None:
        """Append a stage to the pipeline."""

    @abstractmethod
    def insert_stage(self, index: int, stage: StageDefinition) -> None:
        """Insert a stage at a specific position."""

    @abstractmethod
    def remove_stage(self, stage_id: str) -> None:
        """Remove a stage by identifier."""

    @abstractmethod
    async def execute(self, context: PipelineContext) -> None:
        """Run all stages in order against *context*."""


class WorkflowInterface(ABC):
    """Contract for stage-level execution with retries and timeouts."""

    @abstractmethod
    async def execute_stage(
        self,
        stage: StageDefinition,
        input_data: Any,
    ) -> Any:
        """Execute a single stage with retry and timeout logic.

        Parameters
        ----------
        stage:
            The stage to execute.
        input_data:
            Input to pass to *stage.execute*.

        Returns
        -------
        Any
            The output of *stage.execute*.
        """

    @abstractmethod
    async def cancel(self, context: PipelineContext) -> None:
        """Request cancellation for the given execution context."""


class RouterInterface(ABC):
    """Contract for the module registry."""

    @abstractmethod
    def register(self, module_type: str, module: Any) -> None:
        """Register a module instance under *module_type*.

        Parameters
        ----------
        module_type:
            Unique key (e.g. ``"computer_vision"``, ``"friendly"``).
        module:
            The module instance (must implement the corresponding interface).
        """

    @abstractmethod
    def get(self, module_type: str) -> Any:
        """Retrieve a registered module by *module_type*.

        Raises
        ------
        RoutingException
            If the module type has not been registered.
        """

    @abstractmethod
    def registered_types(self) -> list[str]:
        """Return all currently registered module type keys."""
