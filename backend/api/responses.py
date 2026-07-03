"""Standard API response models for every DSS endpoint.

These are runtime-only response types.  No modifications to
``backend.contracts``.
"""

from typing import Any

from pydantic import BaseModel, Field

from backend.contracts.models.analysis import (
    EnemyAnalysis,
    FriendlyAnalysis,
    TerrainAnalysis,
)
from backend.contracts.models.decision import DecisionRecommendation
from backend.contracts.models.detection import DetectionResult
from backend.contracts.models.fusion import FusionResult, ThreatAssessment

# ---------------------------------------------------------------------------
# Generic response envelopes
# ---------------------------------------------------------------------------


class ApiSuccessResponse(BaseModel):
    """Standard success envelope for API responses."""

    success: bool = Field(default=True, description="Indicates the operation succeeded")
    message: str = Field(default="ok", description="Human-readable status message")
    data: dict[str, Any] | None = Field(None, description="Optional response payload")


class ApiErrorResponse(BaseModel):
    """Standard error envelope for API responses."""

    success: bool = Field(default=False, description="Indicates the operation failed")
    error_code: str = Field(default="UNKNOWN", description="Machine-readable error code")
    message: str = Field(
        default="An error occurred", description="Human-readable error description"
    )
    details: dict[str, Any] | None = Field(None, description="Optional structured error details")


# ---------------------------------------------------------------------------
# Domain-specific response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    status: str = Field(default="healthy", description="Service health status")
    service: str = Field(default="DSSPrototype", description="Service name")
    version: str = Field(default="0.1.0", description="Application version")


class SystemInfoResponse(BaseModel):
    """Detailed system information response."""

    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Runtime environment")
    modules_registered: int = Field(..., ge=0, description="Number of registered modules")
    modules: list[str] = Field(default_factory=list, description="Registered module type keys")
    pipeline_ready: bool = Field(..., description="Whether the pipeline is ready for execution")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Configuration summary"
    )


class ExecuteResponse(BaseModel):
    """Response payload for a pipeline execution request."""

    request_id: str = Field(default="", description="Unique request identifier")
    pipeline_id: str = Field(default="", description="Unique pipeline execution identifier")
    status: str = Field(default="unknown", description="Execution status (completed|failed)")
    total_duration_ms: float = Field(default=0.0, ge=0, description="Total execution time in ms")
    stage_durations: dict[str, float] = Field(
        default_factory=dict, description="Per-stage execution times in ms"
    )
    errors: list[str] = Field(default_factory=list, description="Execution error messages")
    warnings: list[str] = Field(default_factory=list, description="Execution warning messages")

    # Stage results (optional — populated on success)
    detection: DetectionResult | None = Field(None, description="Computer Vision detection result")
    friendly: FriendlyAnalysis | None = Field(None, description="Friendly force analysis")
    enemy: EnemyAnalysis | None = Field(None, description="Enemy force analysis")
    terrain: TerrainAnalysis | None = Field(None, description="Terrain analysis")
    fusion: FusionResult | None = Field(None, description="Fused intelligence result")
    threat: ThreatAssessment | None = Field(None, description="Threat assessment")
    decision: DecisionRecommendation | None = Field(
        None, description="Decision recommendation"
    )


class PipelineStatusResponse(BaseModel):
    """Pipeline configuration and execution-order response."""

    status: str = Field(default="configured", description="Pipeline status")
    stage_count: int = Field(default=0, ge=0, description="Number of configured stages")
    execution_order: list[str] = Field(
        default_factory=list, description="Stage execution order"
    )
    parallel_groups: list[list[str]] = Field(
        default_factory=list, description="Stage groups eligible for parallel execution"
    )


class DecisionCapabilitiesResponse(BaseModel):
    """Decision Engine capabilities response."""

    available: bool = Field(..., description="Whether the Decision Engine is available")
    capabilities: list[str] = Field(
        default_factory=list,
        description="List of supported decision capabilities",
    )
    priority_range: str = Field(
        default="1 (highest) to 5 (lowest)",
        description="Priority range supported",
    )
    coa_sources: str = Field(
        default="configurable templates",
        description="Source of Course-of-Action templates",
    )
