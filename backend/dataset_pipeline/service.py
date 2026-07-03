"""PipelineService — public facade for the dataset pipeline.

Exposes individual stage methods and a full pipeline method.
All methods delegate to the underlying workflow and ingestor.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from backend.dataset_pipeline.ingest import DatasetIngestor
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
from backend.dataset_pipeline.workflow import DatasetWorkflow

logger = logging.getLogger("dss.dataset_pipeline.service")


def _single_stage_result(
    dataset_name: str,
    stage: PipelineStageResult,
    context: DatasetContext,
) -> PipelineResult:
    status = PipelineStatus.COMPLETED if stage.status == PipelineStatus.COMPLETED else stage.status
    return PipelineResult(
        status=status,
        dataset_name=dataset_name,
        stages=[stage],
        summary=ExecutionSummary(
            stages_total=1,
            stages_completed=1 if status == PipelineStatus.COMPLETED else 0,
            stages_failed=1 if status == PipelineStatus.FAILED else 0,
        ),
        context=context,
    )


class PipelineService(PipelineServiceInterface):
    """Public facade for the dataset pipeline.

    Usage:
        service = PipelineService()
        result = await service.ingest_dataset("coco2017", Path("/data/coco2017"))
    """

    def __init__(
        self,
        ingestor: DatasetIngestorInterface | None = None,
        workflow: WorkflowInterface | None = None,
    ) -> None:
        self._workflow = workflow or DatasetWorkflow()
        self._ingestor = ingestor or DatasetIngestor(workflow=self._workflow)

    async def ingest_dataset(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        skip_quality: bool = False,
        skip_training: bool = False,
        continue_on_error: bool = False,
        dry_run: bool = False,
        output_dir: Path | None = None,
        force: bool = False,
    ) -> PipelineResult:
        """Ingest a dataset through the entire pipeline."""
        return await self._ingestor.ingest(
            dataset_name=dataset_name,
            source_path=source_path,
            skip_quality=skip_quality,
            skip_training=skip_training,
            continue_on_error=continue_on_error,
            dry_run=dry_run,
            output_dir=output_dir,
            force=force,
        )

    async def run_catalog(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult:
        wf = cast(DatasetWorkflow, self._workflow)
        context = DatasetContext(
            dataset_name=dataset_name,
            source_path=source_path.resolve(),
        )
        stage = await wf._run_catalog_stage(context, dry_run, force=force)
        return _single_stage_result(dataset_name, stage, context)

    async def run_mapping(
        self,
        dataset_name: str,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult:
        wf = cast(DatasetWorkflow, self._workflow)
        context = DatasetContext(
            dataset_name=dataset_name,
            source_path=Path(dataset_name),
        )
        stage = await wf._run_mapping_stage(context, dry_run)
        return _single_stage_result(dataset_name, stage, context)

    async def run_conversion(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult:
        wf = cast(DatasetWorkflow, self._workflow)
        context = DatasetContext(
            dataset_name=dataset_name,
            source_path=source_path.resolve(),
        )
        stage = await wf._run_conversion_stage(context, dry_run)
        return _single_stage_result(dataset_name, stage, context)

    async def run_quality(
        self,
        dataset_name: str,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult:
        wf = cast(DatasetWorkflow, self._workflow)
        context = DatasetContext(
            dataset_name=dataset_name,
            source_path=Path(dataset_name),
        )
        stage = await wf._run_quality_stage(context, dry_run)
        return _single_stage_result(dataset_name, stage, context)

    async def run_training(
        self,
        dataset_name: str,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult:
        wf = cast(DatasetWorkflow, self._workflow)
        context = DatasetContext(
            dataset_name=dataset_name,
            source_path=Path(dataset_name),
        )
        stage = await wf._run_training_stage(context, dry_run)
        return _single_stage_result(dataset_name, stage, context)

    async def run_full_pipeline(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        skip_quality: bool = False,
        skip_training: bool = False,
        continue_on_error: bool = False,
        dry_run: bool = False,
        output_dir: Path | None = None,
        force: bool = False,
    ) -> PipelineResult:
        return await self.ingest_dataset(
            dataset_name=dataset_name,
            source_path=source_path,
            skip_quality=skip_quality,
            skip_training=skip_training,
            continue_on_error=continue_on_error,
            dry_run=dry_run,
            output_dir=output_dir,
            force=force,
        )
