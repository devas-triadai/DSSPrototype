"""DatasetIngestor — validates source paths, detects format,
creates execution context, and delegates to DatasetWorkflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.dataset_pipeline.exceptions import (
    DatasetNotFoundError,
    PipelineValidationError,
)
from backend.dataset_pipeline.interfaces import DatasetIngestorInterface, WorkflowInterface
from backend.dataset_pipeline.models import DatasetContext, PipelineResult, PipelineStatus
from backend.dataset_pipeline.workflow import DatasetWorkflow

logger = logging.getLogger("dss.dataset_pipeline.ingest")


class DatasetIngestor(DatasetIngestorInterface):
    """Handles the initial validation and context creation for dataset ingestion.

    Delegates the actual pipeline execution to DatasetWorkflow.
    """

    def __init__(self, workflow: WorkflowInterface | None = None) -> None:
        self._workflow = workflow or DatasetWorkflow()

    async def validate_source(self, source_path: Path) -> str:
        """Validate the dataset source path and return the detected format."""
        if not source_path.exists():
            raise DatasetNotFoundError(
                dataset_name=source_path.name,
                path=str(source_path),
            )

        source_path = source_path.resolve()

        if source_path.is_dir():
            images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
            if not images:
                raise PipelineValidationError(
                    f"No images found in directory: {source_path}",
                )
        elif source_path.is_file():
            ext = source_path.suffix.lower()
            if ext not in {".json", ".yaml", ".yml", ".csv", ".xml"}:
                raise PipelineValidationError(
                    f"Unsupported dataset file format: {ext}",
                )
        else:
            raise DatasetNotFoundError(
                dataset_name=source_path.name,
                path=str(source_path),
            )

        return self._detect_format(source_path)

    async def ingest(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        skip_quality: bool = False,
        skip_training: bool = False,
        continue_on_error: bool = False,
        dry_run: bool = False,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        """Ingest a dataset through the entire pipeline."""
        logger.info(
            "Ingestion started | dataset=%s | source=%s | dry_run=%s",
            dataset_name, source_path, dry_run,
        )

        source_path = source_path.resolve()
        if not source_path.exists():
            raise DatasetNotFoundError(dataset_name, str(source_path))

        dataset_type = await self.validate_source(source_path)

        context = DatasetContext(
            dataset_name=dataset_name,
            source_path=source_path,
            dataset_type=dataset_type,
        )

        result = await self._workflow.execute(
            context=context,
            skip_quality=skip_quality,
            skip_training=skip_training,
            continue_on_error=continue_on_error,
            dry_run=dry_run,
        )

        if result.status == PipelineStatus.COMPLETED:
            logger.info(
                "SUCCESS  Pipeline completed | dataset=%s | stages=%d/%d",
                dataset_name,
                result.summary.stages_completed,
                result.summary.stages_total,
            )
        else:
            logger.warning(
                "Pipeline finished with status=%s | dataset=%s | errors=%s",
                result.status, dataset_name, result.error,
            )

        return result

    @staticmethod
    def _detect_format(source_path: Path) -> str:
        name_lower = source_path.name.lower()
        if "coco" in name_lower:
            return "coco_json"
        if "yolo" in name_lower or "darknet" in name_lower:
            return "yolo_txt"
        if "voc" in name_lower or "pascal" in name_lower:
            return "pascal_voc"
        if source_path.is_file() and source_path.suffix.lower() == ".json":
            return "coco_json"
        if source_path.is_file() and source_path.suffix.lower() in {".yaml", ".yml"}:
            return "yolo_txt"
        return "coco_json"
