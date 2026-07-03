"""Runtime health checks for the DSS pipeline.

Verifies registered modules, dependency graph, configuration,
and pipeline readiness.  No external services required.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from backend.modules.orchestration.dependency_resolver import DependencyResolver
from backend.modules.orchestration.module_registry import ModuleRegistry

logger = logging.getLogger("dss.orchestration.health")


@dataclass(frozen=True)
class HealthStatus:
    """Immutable health status for the DSS runtime."""

    healthy: bool = False
    modules_registered: int = 0
    modules: list[str] = field(default_factory=list)
    missing_modules: list[str] = field(default_factory=list)
    dependency_issues: list[str] = field(default_factory=list)
    pipeline_ready: bool = False
    config: dict[str, Any] = field(default_factory=dict)


class RuntimeHealth:
    """Health checks for the Runtime Integration Layer.

    Inspects module registry completeness, dependency graph
    consistency, and pipeline configuration readiness.
    """

    def __init__(
        self,
        registry: ModuleRegistry,
        resolver: DependencyResolver,
        config: Any | None = None,
    ) -> None:
        self._registry = registry
        self._resolver = resolver
        self._config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> HealthStatus:
        """Run all health checks and return the aggregated status.

        Returns
        -------
        HealthStatus
            Frozen status object with check results.
        """
        registered = self._registry.registered_types()
        missing = self._registry.missing_modules()
        dep_issues = self._resolver.verify_connections()

        pipeline_ready = not missing and not dep_issues

        config_info: dict[str, Any] = {}
        if self._config:
            config_info = {
                "pipeline_timeout_seconds": getattr(
                    self._config, "pipeline_timeout_seconds", None
                ),
                "stage_timeout_seconds": getattr(
                    self._config, "default_stage_timeout_seconds", None
                ),
                "retry_count": getattr(
                    self._config, "default_retry_count", None
                ),
            }

        return HealthStatus(
            healthy=pipeline_ready,
            modules_registered=len(registered),
            modules=registered,
            missing_modules=missing,
            dependency_issues=dep_issues,
            pipeline_ready=pipeline_ready,
            config=config_info,
        )

    def summary(self) -> str:
        """Return a human-readable health summary string."""
        status = self.check()
        if status.healthy:
            return (
                f"Runtime healthy: {status.modules_registered} modules registered, "
                f"pipeline ready"
            )
        parts: list[str] = ["Runtime degraded"]

        if status.missing_modules:
            parts.append(f"missing={status.missing_modules}")
        if status.dependency_issues:
            parts.append(f"deps={status.dependency_issues}")

        return "; ".join(parts)
