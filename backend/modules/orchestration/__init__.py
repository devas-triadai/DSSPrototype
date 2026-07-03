"""Orchestration Engine — pipeline coordinator.

Coordinates the end-to-end intelligence processing pipeline.
Contains no business logic, AI, or module-specific code.
"""

from backend.modules.orchestration.config import OrchestrationConfig, orch_config
from backend.modules.orchestration.context import PipelineContext
from backend.modules.orchestration.dependency_resolver import (
    DependencyGraph,
    DependencyResolver,
)
from backend.modules.orchestration.exceptions import (
    CancellationError,
    PipelineError,
    RoutingError,
    StateTransitionError,
    TimeoutError,
    WorkflowError,
)
from backend.modules.orchestration.health import HealthStatus, RuntimeHealth
from backend.modules.orchestration.interfaces import (
    PipelineInterface,
    RouterInterface,
    StageDefinition,
    WorkflowInterface,
)
from backend.modules.orchestration.metrics import MetricsCollector, RuntimeMetrics, StageMetrics
from backend.modules.orchestration.module_registry import ModuleRegistry
from backend.modules.orchestration.pipeline import Pipeline
from backend.modules.orchestration.pipeline_executor import PipelineExecutor
from backend.modules.orchestration.result_assembler import (
    ExecutionMetadata,
    ResultAssembler,
    RuntimeResult,
)
from backend.modules.orchestration.router import Router
from backend.modules.orchestration.runtime import Runtime
from backend.modules.orchestration.service import OrchestrationService
from backend.modules.orchestration.state import PipelineState, StateMachine
from backend.modules.orchestration.workflow import Workflow

__all__ = [
    # Config
    "OrchestrationConfig",
    "orch_config",
    # Original orchestration
    "OrchestrationService",
    "Pipeline",
    "PipelineContext",
    "PipelineState",
    "StateMachine",
    "StageDefinition",
    "Router",
    "Workflow",
    "PipelineInterface",
    "RouterInterface",
    "WorkflowInterface",
    "PipelineError",
    "WorkflowError",
    "RoutingError",
    "TimeoutError",
    "CancellationError",
    "StateTransitionError",
    # Runtime Integration Layer
    "Runtime",
    "PipelineExecutor",
    "ModuleRegistry",
    "DependencyResolver",
    "DependencyGraph",
    "ResultAssembler",
    "RuntimeResult",
    "ExecutionMetadata",
    "MetricsCollector",
    "RuntimeMetrics",
    "StageMetrics",
    "RuntimeHealth",
    "HealthStatus",
]
