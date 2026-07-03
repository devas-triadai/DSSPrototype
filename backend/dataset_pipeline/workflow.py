"""DatasetWorkflow — stage orchestration for dataset ingestion.

Coordinates CatalogService → OntologyMappingService →
DatasetConversionService → DatasetQualityService → TrainingService.

No business logic — only orchestration.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

from backend.dataset_catalog.models import CatalogEntry
from backend.dataset_catalog.service import DatasetCatalogService
from backend.dataset_conversion.models import CanonicalDataset, LoadResult
from backend.dataset_conversion.service import DatasetConversionService
from backend.dataset_pipeline.config import pipeline_config
from backend.dataset_pipeline.exceptions import StageExecutionError
from backend.dataset_pipeline.interfaces import WorkflowInterface
from backend.dataset_pipeline.models import (
    DatasetContext,
    ExecutionSummary,
    PipelineResult,
    PipelineStageResult,
    PipelineStatus,
)
from backend.dataset_quality.models import QualityReport
from backend.dataset_quality.service import DatasetQualityService
from backend.ontology_mapping.models import DatasetProfile as OntoDatasetProfile
from backend.ontology_mapping.models import MappingResult
from backend.ontology_mapping.service import OntologyMappingService
from backend.training.models import TrainingConfigData, TrainingResult
from backend.training.service import TrainingService

logger = logging.getLogger("dss.dataset_pipeline.workflow")


class DatasetWorkflow(WorkflowInterface):
    """Orchestrates the five stages of dataset ingestion.

    Each stage delegates to an existing service. No business logic
    is duplicated — only coordination and result passing.
    """

    def __init__(
        self,
        catalog_service: DatasetCatalogService | None = None,
        ontology_service: OntologyMappingService | None = None,
        conversion_service: DatasetConversionService | None = None,
        quality_service: DatasetQualityService | None = None,
        training_service: TrainingService | None = None,
    ) -> None:
        self._catalog = catalog_service or DatasetCatalogService()
        self._ontology = ontology_service or OntologyMappingService()
        self._conversion = conversion_service or DatasetConversionService()
        self._quality = quality_service or DatasetQualityService()
        self._training = training_service or TrainingService()

    async def execute(
        self,
        context: DatasetContext,
        skip_quality: bool = False,
        skip_training: bool = False,
        continue_on_error: bool = False,
        dry_run: bool = False,
    ) -> PipelineResult:
        """Execute the full pipeline against the given context."""
        stages: list[PipelineStageResult] = []
        overall_status = PipelineStatus.RUNNING
        overall_error: str | None = None
        start_time = time.monotonic()

        try:
            s1 = await self._run_catalog_stage(context, dry_run)
            stages.append(s1)
            if self._should_stop(s1.status, continue_on_error):
                overall_status = s1.status
                overall_error = s1.error
                return self._build_result(
                    context, stages, overall_status, overall_error, start_time,
                )

            s2 = await self._run_mapping_stage(context, dry_run)
            stages.append(s2)
            if self._should_stop(s2.status, continue_on_error):
                overall_status = s2.status
                overall_error = s2.error
                return self._build_result(
                    context, stages, overall_status, overall_error, start_time,
                )

            s3 = await self._run_conversion_stage(context, dry_run)
            stages.append(s3)
            if self._should_stop(s3.status, continue_on_error):
                overall_status = s3.status
                overall_error = s3.error
                return self._build_result(
                    context, stages, overall_status, overall_error, start_time,
                )

            if not skip_quality:
                s4 = await self._run_quality_stage(context, dry_run)
                stages.append(s4)
                if self._should_stop(s4.status, continue_on_error):
                    overall_status = s4.status
                    overall_error = s4.error
                    return self._build_result(
                        context, stages, overall_status, overall_error, start_time,
                    )
            else:
                stages.append(
                    PipelineStageResult(stage="quality", status=PipelineStatus.SKIPPED)
                )

            if not skip_training:
                s5 = await self._run_training_stage(context, dry_run)
                stages.append(s5)
                if self._should_stop(s5.status, continue_on_error):
                    overall_status = s5.status
                    overall_error = s5.error
                    return self._build_result(
                        context, stages, overall_status, overall_error, start_time,
                    )
            else:
                stages.append(
                    PipelineStageResult(stage="training", status=PipelineStatus.SKIPPED)
                )

            overall_status = PipelineStatus.COMPLETED

        except StageExecutionError as exc:
            logger.error("Pipeline aborted at stage %s: %s", exc.stage, exc)
            overall_status = PipelineStatus.FAILED
            overall_error = str(exc)
            stages.append(
                PipelineStageResult(
                    stage=exc.stage,
                    status=PipelineStatus.FAILED,
                    error=str(exc),
                )
            )

        return self._build_result(context, stages, overall_status, overall_error, start_time)

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_catalog_stage(
        self, context: DatasetContext, dry_run: bool,
    ) -> PipelineStageResult:
        logger.info("Registering dataset in catalog...")
        stage_start = time.monotonic()
        try:
            if dry_run:
                return PipelineStageResult(
                    stage="catalog",
                    status=PipelineStatus.SKIPPED,
                    details={"dry_run": True},
                )

            entry: CatalogEntry = await asyncio.to_thread(
                self._catalog.discover,
                path=context.source_path,
                source_id=pipeline_config.catalog_source_id,
                source_type=pipeline_config.catalog_source_type,
                curator=pipeline_config.catalog_curator,
            )

            context.catalog_entry_id = entry.entry_id
            context.metadata["catalog_entry"] = entry.model_dump()
            context.dataset_type = entry.source_type

            logger.info("Catalog entry created: %s", entry.entry_id)
            return PipelineStageResult(
                stage="catalog",
                status=PipelineStatus.COMPLETED,
                result={"entry_id": entry.entry_id, "dataset_name": entry.name},
                duration_seconds=time.monotonic() - stage_start,
            )
        except Exception as exc:
            logger.error("Catalog stage failed: %s", exc)
            return PipelineStageResult(
                stage="catalog",
                status=PipelineStatus.FAILED,
                error=str(exc),
                duration_seconds=time.monotonic() - stage_start,
            )

    async def _run_mapping_stage(
        self, context: DatasetContext, dry_run: bool,
    ) -> PipelineStageResult:
        logger.info("Mapping ontology...")
        stage_start = time.monotonic()
        try:
            if dry_run:
                return PipelineStageResult(
                    stage="mapping",
                    status=PipelineStatus.SKIPPED,
                    details={"dry_run": True},
                )

            labels = list(context.metadata.get("labels", []))
            if not labels and context.metadata.get("catalog_entry"):
                profile = context.metadata["catalog_entry"].get("profile", {})
                classes = profile.get("classes", [])
                labels = [str(c) for c in classes] if classes else [context.dataset_name]

            onto_labels = tuple(labels) if labels else (context.dataset_name,)
            onto_profile = OntoDatasetProfile(
                dataset_name=context.dataset_name,
                label_count=len(onto_labels),
                labels=onto_labels,
                version="1.0.0",
                description=f"Pipeline-ingested dataset: {context.dataset_name}",
            )

            await self._ontology.register_dataset(onto_profile, [])
            results: list[MappingResult] = await self._ontology.map_dataset(
                context.dataset_name, labels,
            )

            context.metadata["mapping_results"] = [r.model_dump() for r in results]
            logger.info("Ontology mapping complete: %d labels mapped", len(results))
            return PipelineStageResult(
                stage="mapping",
                status=PipelineStatus.COMPLETED,
                result={
                    "labels_mapped": len(results),
                    "results": [r.model_dump() for r in results],
                },
                duration_seconds=time.monotonic() - stage_start,
            )
        except Exception as exc:
            logger.error("Mapping stage failed: %s", exc)
            return PipelineStageResult(
                stage="mapping",
                status=PipelineStatus.FAILED,
                error=str(exc),
                duration_seconds=time.monotonic() - stage_start,
            )

    async def _run_conversion_stage(
        self, context: DatasetContext, dry_run: bool,
    ) -> PipelineStageResult:
        logger.info("Converting dataset...")
        stage_start = time.monotonic()
        try:
            if dry_run:
                return PipelineStageResult(
                    stage="conversion",
                    status=PipelineStatus.SKIPPED,
                    details={"dry_run": True},
                )

            output_path = str(
                Path(pipeline_config.conversion_output_dir) / context.dataset_name
            )

            source_format = context.dataset_type or self._detect_format(context.source_path)

            load_result: LoadResult = await self._conversion.load_dataset(
                path=str(context.source_path),
                dataset_format=source_format,
            )

            canonical: CanonicalDataset = await self._conversion.convert_dataset(
                load_result=load_result,
                dataset_name=context.dataset_name,
            )

            export_result = await self._conversion.export_dataset(
                dataset=canonical,
                export_format=pipeline_config.conversion_output_format,
                output_path=output_path,
            )

            context.canonical_dataset = canonical
            context.metadata["conversion_export"] = export_result.model_dump()
            context.metadata["canonical_dataset_id"] = canonical.id

            logger.info(
                "Conversion complete: %d images, %d annotations",
                canonical.image_count, canonical.annotation_count,
            )
            return PipelineStageResult(
                stage="conversion",
                status=PipelineStatus.COMPLETED,
                result={
                    "dataset_id": canonical.id,
                    "images": canonical.image_count,
                    "annotations": canonical.annotation_count,
                    "classes": canonical.class_count,
                    "export_path": output_path,
                },
                duration_seconds=time.monotonic() - stage_start,
            )
        except Exception as exc:
            logger.error("Conversion stage failed: %s", exc)
            return PipelineStageResult(
                stage="conversion",
                status=PipelineStatus.FAILED,
                error=str(exc),
                duration_seconds=time.monotonic() - stage_start,
            )

    async def _run_quality_stage(
        self, context: DatasetContext, dry_run: bool,
    ) -> PipelineStageResult:
        logger.info("Running quality checks...")
        stage_start = time.monotonic()
        try:
            if dry_run:
                return PipelineStageResult(
                    stage="quality",
                    status=PipelineStatus.SKIPPED,
                    details={"dry_run": True},
                )

            if context.canonical_dataset is None:
                raise StageExecutionError(
                    "quality",
                    "No canonical dataset available — run conversion first.",
                )

            image_dir = pipeline_config.quality_image_dir or str(context.source_path)

            report: QualityReport = await self._quality.run_pipeline(
                dataset=context.canonical_dataset,
                image_dir=image_dir,
            )

            context.quality_report = report
            passed = report.overall_score.production_ready

            grade = report.overall_score.letter_grade
            if hasattr(grade, 'value'):
                grade_str = grade.value
            else:
                grade_str = str(grade)

            logger.info(
                "Quality checks complete: score=%.2f, grade=%s, passed=%s",
                report.overall_score.overall,
                grade_str,
                passed,
            )
            return PipelineStageResult(
                stage="quality",
                status=PipelineStatus.COMPLETED,
                result={
                    "quality_score": report.overall_score.overall,
                    "letter_grade": str(report.overall_score.letter_grade),
                    "production_ready": passed,
                    "errors": report.error_count,
                    "warnings": report.warning_count,
                },
                duration_seconds=time.monotonic() - stage_start,
            )
        except Exception as exc:
            logger.error("Quality stage failed: %s", exc)
            return PipelineStageResult(
                stage="quality",
                status=PipelineStatus.FAILED,
                error=str(exc),
                duration_seconds=time.monotonic() - stage_start,
            )

    async def _run_training_stage(
        self, context: DatasetContext, dry_run: bool,
    ) -> PipelineStageResult:
        logger.info("Starting training...")
        stage_start = time.monotonic()
        try:
            if dry_run:
                return PipelineStageResult(
                    stage="training",
                    status=PipelineStatus.SKIPPED,
                    details={"dry_run": True},
                )

            config = TrainingConfigData(
                model_name=pipeline_config.training_model_name,
                dataset_version=context.dataset_name,
                experiment_name=f"pipeline_{context.dataset_name}",
                batch_size=pipeline_config.training_batch_size,
                epochs=pipeline_config.training_epochs,
                learning_rate=pipeline_config.training_learning_rate,
            )

            result: TrainingResult = await asyncio.to_thread(
                self._training.run_pipeline, config,
            )

            context.training_result = result

            logger.info(
                "Training complete: experiment=%s, model=%s, status=%s",
                result.experiment_id, result.model_id, result.status,
            )
            return PipelineStageResult(
                stage="training",
                status=PipelineStatus.COMPLETED,
                result={
                    "experiment_id": result.experiment_id,
                    "model_id": result.model_id,
                    "epochs_completed": result.total_epochs_completed,
                    "best_metric": result.best_metric,
                    "status": result.status,
                },
                duration_seconds=time.monotonic() - stage_start,
            )
        except Exception as exc:
            logger.error("Training stage failed: %s", exc)
            return PipelineStageResult(
                stage="training",
                status=PipelineStatus.FAILED,
                error=str(exc),
                duration_seconds=time.monotonic() - stage_start,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _should_stop(status: PipelineStatus, continue_on_error: bool) -> bool:
        if status == PipelineStatus.COMPLETED or status == PipelineStatus.SKIPPED:
            return False
        if status == PipelineStatus.FAILED and continue_on_error:
            return False
        return True

    @staticmethod
    def _detect_format(source_path: Path) -> str:
        """Heuristically detect the dataset format from the path."""
        name_lower = source_path.name.lower()
        if "coco" in name_lower:
            return "coco_json"
        if "yolo" in name_lower or "darknet" in name_lower:
            return "yolo_txt"
        if "voc" in name_lower or "pascal" in name_lower:
            return "pascal_voc"
        return "coco_json"

    @staticmethod
    def _build_result(
        context: DatasetContext,
        stages: list[PipelineStageResult],
        status: PipelineStatus,
        error: str | None,
        start_time: float,
    ) -> PipelineResult:
        duration = time.monotonic() - start_time
        completed = sum(1 for s in stages if s.status == PipelineStatus.COMPLETED)
        failed = sum(1 for s in stages if s.status == PipelineStatus.FAILED)
        skipped = sum(1 for s in stages if s.status == PipelineStatus.SKIPPED)

        summary = ExecutionSummary(
            total_duration_seconds=round(duration, 3),
            stages_completed=completed,
            stages_failed=failed,
            stages_skipped=skipped,
            stages_total=len(stages),
        )

        return PipelineResult(
            status=status,
            dataset_name=context.dataset_name,
            stages=stages,
            summary=summary,
            context=context,
            error=error,
        )
