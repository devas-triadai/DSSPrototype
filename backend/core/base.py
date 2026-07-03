"""Abstract base classes for the modular architecture.

Every AI capability module **must** inherit from ``BaseModule``
to guarantee a consistent lifecycle (init / shutdown) and allow
the Orchestrator to manage them uniformly.
"""

from abc import ABC, abstractmethod


class BaseModule(ABC):
    """Template for every DSS AI module.

    Subclasses implement ``initialize`` and ``shutdown``.
    The ``name`` attribute serves as a unique identifier.

    Usage::

        class MyAgent(BaseModule):
            def __init__(self) -> None:
                super().__init__(name="my_agent")

            def initialize(self) -> None:
                ...   # load model, connect to DB, etc.

            def shutdown(self) -> None:
                ...   # release resources
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """Allocate resources, load models, connect to services."""
        self._initialized = True

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources, close connections, save state."""
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Return ``True`` if the module has been initialized."""
        return self._initialized

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
