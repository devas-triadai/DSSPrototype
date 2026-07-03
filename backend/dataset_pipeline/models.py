"""Pipeline data models.

PipelineResult, PipelineStageResult, PipelineStatus, ExecutionSummary,
and DatasetContext for tracking pipeline execution.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineStageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stage: str
    status: PipelineStatus
    result: Any = None
    error: str | None = None
    duration_seconds: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)


class ExecutionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_duration_seconds: float = 0.0
    stages_completed: int = 0
    stages_failed: int = 0
    stages_skipped: int = 0
    stages_total: int = 0


class DatasetContext(BaseModel):
    model_config = ConfigDict(frozen=False)

    dataset_name: str
    source_path: Path
    dataset_type: str = ""
    catalog_entry_id: str | None = None
    canonical_dataset: Any = None
    quality_report: Any = None
    training_result: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: PipelineStatus
    dataset_name: str
    stages: list[PipelineStageResult] = Field(default_factory=list)
    summary: ExecutionSummary = Field(default_factory=ExecutionSummary)
    context: DatasetContext | None = None
    error: str | None = None
