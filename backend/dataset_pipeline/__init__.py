"""Dataset Pipeline — orchestration layer for dataset ingestion.

Coordinates Dataset Catalog → Ontology Mapping → Dataset Conversion →
Dataset Quality → Training as a single, unified workflow.
"""

from backend.dataset_pipeline.config import PipelineConfig, pipeline_config
from backend.dataset_pipeline.exceptions import (
    DatasetNotFoundError,
    PipelineError,
    PipelineValidationError,
    StageExecutionError,
)
from backend.dataset_pipeline.interfaces import (
    DatasetIngestorInterface,
    PipelineServiceInterface,
    WorkflowInterface,
)
from backend.dataset_pipeline.models import (
    DatasetContext,
    ExecutionSummary,
    PipelineResult,
    PipelineStageResult,
    PipelineStatus,
)
from backend.dataset_pipeline.service import PipelineService
from backend.dataset_pipeline.workflow import DatasetWorkflow

__all__ = [
    "PipelineConfig",
    "pipeline_config",
    "DatasetNotFoundError",
    "PipelineError",
    "PipelineValidationError",
    "StageExecutionError",
    "DatasetIngestorInterface",
    "PipelineServiceInterface",
    "WorkflowInterface",
    "DatasetContext",
    "ExecutionSummary",
    "PipelineResult",
    "PipelineStageResult",
    "PipelineStatus",
    "PipelineService",
    "DatasetWorkflow",
]
