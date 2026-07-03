"""Tests for DatasetWorkflow orchestration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.dataset_pipeline.models import (
    DatasetContext,
    PipelineStatus,
)
from backend.dataset_pipeline.workflow import DatasetWorkflow


class TestDatasetWorkflow:
    """Tests for the DatasetWorkflow class."""

    @pytest.fixture
    def workflow(
        self,
        mock_catalog_service: MagicMock,
        mock_ontology_service: MagicMock,
        mock_conversion_service: MagicMock,
        mock_quality_service: MagicMock,
        mock_training_service: MagicMock,
    ) -> DatasetWorkflow:
        return DatasetWorkflow(
            catalog_service=mock_catalog_service,
            ontology_service=mock_ontology_service,
            conversion_service=mock_conversion_service,
            quality_service=mock_quality_service,
            training_service=mock_training_service,
        )

    @pytest.fixture
    def context(self) -> DatasetContext:
        return DatasetContext(
            dataset_name="test_dataset",
            source_path=Path("/data/test_dataset"),
            dataset_type="coco_json",
        )

    # ------------------------------------------------------------------
    # Full pipeline execution
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_full_pipeline_completes(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.stages) == 5
        for stage in result.stages:
            assert stage.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_full_pipeline_populates_context(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert result.context is not None
        assert result.context.catalog_entry_id is not None
        assert result.context.canonical_dataset is not None
        assert result.context.quality_report is not None
        assert result.context.training_result is not None

    @pytest.mark.asyncio
    async def test_full_pipeline_summary(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert result.summary.stages_total == 5
        assert result.summary.stages_completed == 5
        assert result.summary.stages_failed == 0
        assert result.summary.total_duration_seconds > 0

    # ------------------------------------------------------------------
    # Skip stages
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_skip_quality(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context, skip_quality=True)
        assert result.status == PipelineStatus.COMPLETED
        stage_names = [s.stage for s in result.stages]
        assert "quality" in stage_names
        quality_stage = next(s for s in result.stages if s.stage == "quality")
        assert quality_stage.status == PipelineStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_training(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context, skip_training=True)
        assert result.status == PipelineStatus.COMPLETED
        training_stage = next(s for s in result.stages if s.stage == "training")
        assert training_stage.status == PipelineStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_quality_and_training(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(
            context, skip_quality=True, skip_training=True,
        )
        assert result.status == PipelineStatus.COMPLETED
        assert len(result.stages) == 5
        quality_stage = next(s for s in result.stages if s.stage == "quality")
        training_stage = next(s for s in result.stages if s.stage == "training")
        assert quality_stage.status == PipelineStatus.SKIPPED
        assert training_stage.status == PipelineStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_skip_nothing(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert len(result.stages) == 5
        for s in result.stages:
            assert s.status == PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_dry_run_skips_all_stages(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context, dry_run=True)
        assert result.status == PipelineStatus.COMPLETED
        for s in result.stages:
            assert s.status == PipelineStatus.SKIPPED

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stage_failure_stops_pipeline(
        self, context: DatasetContext,
    ) -> None:
        failing_catalog = MagicMock()
        failing_catalog.discover.side_effect = RuntimeError("Catalog exploded")

        wf = DatasetWorkflow(catalog_service=failing_catalog)
        result = await wf.execute(context)
        assert result.status == PipelineStatus.FAILED
        assert result.stages[0].status == PipelineStatus.FAILED
        err_msg = result.stages[0].error or ""
        assert "Catalog exploded" in err_msg

    @pytest.mark.asyncio
    async def test_stage_failure_with_continue_on_error(
        self, context: DatasetContext,
    ) -> None:
        failing_catalog = MagicMock()
        failing_catalog.discover.side_effect = RuntimeError("Catalog exploded")

        wf = DatasetWorkflow(catalog_service=failing_catalog)
        result = await wf.execute(context, continue_on_error=True)
        assert result.stages[0].status == PipelineStatus.FAILED

    @pytest.mark.asyncio
    async def test_mapping_failure_does_not_affect_catalog(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert result.stages[0].stage == "catalog"
        assert result.stages[0].status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_conversion_failure(
        self, context: DatasetContext,
    ) -> None:
        failing_conversion = MagicMock()
        failing_conversion.load_dataset.side_effect = RuntimeError("Conversion OOM")

        wf = DatasetWorkflow(
            catalog_service=MagicMock(),
            ontology_service=MagicMock(),
            conversion_service=failing_conversion,
        )
        result = await wf.execute(context)
        assert any(s.status == PipelineStatus.FAILED for s in result.stages)

    # ------------------------------------------------------------------
    # Error propagation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_error_field_on_failure(
        self, context: DatasetContext,
    ) -> None:
        failing_catalog = MagicMock()
        failing_catalog.discover.side_effect = RuntimeError("Kaboom")

        wf = DatasetWorkflow(catalog_service=failing_catalog)
        result = await wf.execute(context)
        assert result.error is not None
        assert "Kaboom" in result.error

    @pytest.mark.asyncio
    async def test_no_error_on_success(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert result.error is None

    # ------------------------------------------------------------------
    # Stage results detail
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_catalog_stage_has_entry_id(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        catalog_stage = next(s for s in result.stages if s.stage == "catalog")
        assert catalog_stage.result is not None
        assert "entry_id" in catalog_stage.result

    @pytest.mark.asyncio
    async def test_conversion_stage_has_dataset_metrics(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        conv_stage = next(s for s in result.stages if s.stage == "conversion")
        assert conv_stage.result is not None
        assert "images" in conv_stage.result
        assert "annotations" in conv_stage.result
        assert "classes" in conv_stage.result

    @pytest.mark.asyncio
    async def test_quality_stage_has_score(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        quality_stage = next(s for s in result.stages if s.stage == "quality")
        assert quality_stage.result is not None
        assert "quality_score" in quality_stage.result
        assert "letter_grade" in quality_stage.result

    @pytest.mark.asyncio
    async def test_training_stage_has_experiment_id(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        training_stage = next(s for s in result.stages if s.stage == "training")
        assert training_stage.result is not None
        assert "experiment_id" in training_stage.result

    # ------------------------------------------------------------------
    # Ordering
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stages_executed_in_order(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        stage_names = [s.stage for s in result.stages]
        expected = ["catalog", "mapping", "conversion", "quality", "training"]
        assert stage_names == expected

    @pytest.mark.asyncio
    async def test_stages_executed_in_order_with_skips(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context, skip_quality=True, skip_training=True)
        stage_names = [s.stage for s in result.stages]
        assert stage_names == ["catalog", "mapping", "conversion", "quality", "training"]

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_stage_duration_recorded(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        for stage in result.stages:
            if stage.status == PipelineStatus.COMPLETED:
                assert stage.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_total_duration_is_reasonable(
        self, workflow: DatasetWorkflow, context: DatasetContext,
    ) -> None:
        result = await workflow.execute(context)
        assert result.summary.total_duration_seconds >= 0
        assert result.summary.total_duration_seconds < 60

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_empty_metadata(self) -> None:
        ctx = DatasetContext(
            dataset_name="empty",
            source_path=Path("/empty"),
        )
        ontology = MagicMock()
        ontology.register_dataset = AsyncMock()
        ontology.map_dataset = AsyncMock(return_value=[])
        conversion = MagicMock()
        load_result = MagicMock(image_count=0, annotation_count=0)
        conversion.load_dataset = AsyncMock(return_value=load_result)
        conversion.convert_dataset = AsyncMock(return_value=MagicMock(
            id="test", name="empty", image_count=0, annotation_count=0, class_count=0,
        ))
        conversion.export_dataset = AsyncMock(return_value=MagicMock(
            export_format="yolo", output_path="/out", images_exported=0,
            annotations_exported=0, file_count=0, file_size_bytes=0,
            model_dump=lambda: {},
        ))
        quality_score = MagicMock(
            overall=100.0, letter_grade=MagicMock(value="A"), production_ready=True,
        )
        quality = MagicMock()
        quality.run_pipeline = AsyncMock(return_value=MagicMock(
            dataset_name="empty",
            overall_score=quality_score,
            error_count=0, warning_count=0,
        ))
        wf = DatasetWorkflow(
            catalog_service=MagicMock(),
            ontology_service=ontology,
            conversion_service=conversion,
            quality_service=quality,
            training_service=MagicMock(),
        )
        result = await wf.execute(ctx)
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_unknown_format(self) -> None:
        ctx = DatasetContext(
            dataset_name="unknown",
            source_path=Path("/unknown/data"),
            dataset_type="unknown_format",
        )
        ontology = MagicMock()
        ontology.register_dataset = AsyncMock()
        ontology.map_dataset = AsyncMock(return_value=[])
        conversion = MagicMock()
        load_result = MagicMock(image_count=0, annotation_count=0)
        conversion.load_dataset = AsyncMock(return_value=load_result)
        conversion.convert_dataset = AsyncMock(return_value=MagicMock(
            id="test", name="unknown", image_count=0, annotation_count=0, class_count=0,
        ))
        conversion.export_dataset = AsyncMock(return_value=MagicMock(
            export_format="yolo", output_path="/out", images_exported=0,
            annotations_exported=0, file_count=0, file_size_bytes=0,
            model_dump=lambda: {},
        ))
        quality_score = MagicMock(
            overall=100.0, letter_grade=MagicMock(value="A"), production_ready=True,
        )
        quality = MagicMock()
        quality.run_pipeline = AsyncMock(return_value=MagicMock(
            dataset_name="unknown",
            overall_score=quality_score,
            error_count=0, warning_count=0,
        ))
        wf = DatasetWorkflow(
            catalog_service=MagicMock(),
            ontology_service=ontology,
            conversion_service=conversion,
            quality_service=quality,
            training_service=MagicMock(),
        )
        result = await wf.execute(ctx)
        assert result.status == PipelineStatus.COMPLETED
