"""Dependency resolution for the Runtime Integration Layer.

Builds runtime dependencies, injects modules, and verifies
that all required modules exist before pipeline execution.
"""

import logging
from dataclasses import dataclass, field

from backend.modules.orchestration.module_registry import ModuleRegistry

logger = logging.getLogger("dss.orchestration.dependency_resolver")

# Execution order with explicit dependencies for each pipeline stage.
# Used to validate that all transitive dependencies are registered.
_DEPENDENCY_EDGES: dict[str, list[str]] = {
    "computer_vision": [],
    "friendly": ["computer_vision"],
    "enemy": ["computer_vision"],
    "terrain": ["computer_vision"],
    "fusion": ["friendly", "enemy", "terrain"],
    "decision": ["fusion"],
}

_EXECUTION_ORDER: tuple[str, ...] = (
    "computer_vision",
    "friendly",
    "enemy",
    "terrain",
    "fusion",
    "decision",
)


@dataclass(frozen=True)
class DependencyGraph:
    """Immutable representation of the module dependency structure.

    Attributes
    ----------
    edges:
        Mapping from module type to its prerequisite module types.
    execution_order:
        Topologically-sorted execution order of all stages.
    """

    edges: dict[str, list[str]] = field(
        default_factory=lambda: dict(_DEPENDENCY_EDGES)
    )
    execution_order: tuple[str, ...] = _EXECUTION_ORDER


class DependencyResolver:
    """Resolves and validates module dependencies for the runtime.

    Verifies every required module is registered before pipeline
    execution and checks that the dependency graph is consistent.
    """

    def __init__(self, registry: ModuleRegistry) -> None:
        self._registry = registry
        self._graph = DependencyGraph()

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> list[str]:
        """Verify all required modules are registered.

        Returns
        -------
        list[str]
            Missing module type keys (empty list if complete).
        """
        return self._registry.missing_modules()

    def verify_connections(self) -> list[str]:
        """Verify every registered module's dependencies are met.

        Only validates entries that correspond to separately
        registered modules — pipeline sub-stages such as
        ``"threat"`` (handled by the Fusion module) are
        excluded from module-level validation.

        Returns
        -------
        list[str]
            List of issue descriptions (empty if all clear).
        """
        issues: list[str] = []
        registered = self._registry.registered_types()

        for module_type in self._graph.execution_order:
            if module_type not in registered:
                continue

            for dep in self._graph.edges.get(module_type, []):
                if dep not in registered:
                    issues.append(
                        f"Module '{module_type}' depends on '{dep}' "
                        f"which is not registered"
                    )

        return issues

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> DependencyGraph:
        """Return the immutable dependency graph."""
        return self._graph

    def execution_order(self) -> tuple[str, ...]:
        """Return the topological execution order of stages."""
        return self._graph.execution_order
