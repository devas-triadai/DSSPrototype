"""Tests for pipeline idempotency — duplicate catalog entries.

Verifies that re-running the same dataset does not fail when the
Dataset Catalog raises ``CatalogError("Entry already exists: ...")``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.dataset_catalog.exceptions import CatalogError
from backend.dataset_pipeline.models import DatasetContext, PipelineStatus
from backend.dataset_pipeline.workflow import DatasetWorkflow

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_catalog_error(entry_id: str = "pipeline_test") -> CatalogError:
    return CatalogError(f"Entry already exists: {entry_id}")


def _make_workflow(
    catalog_discover: MagicMock,
    *,
    with_full_mocks: bool = True,
) -> DatasetWorkflow:
    """Create a DatasetWorkflow with full service mocks for pipeline tests."""
    svc = MagicMock()
    svc.discover = catalog_discover

    if with_full_mocks:
        ontology = MagicMock()
        ontology.register_dataset = AsyncMock()
        ontology.map_dataset = AsyncMock(
            return_value=[
                MagicMock(
                    source_label="car",
                    canonical_value="vehicle.car",
                    canonical_name="Car",
                    confidence=0.95,
                    model_dump=lambda: {"source_label": "car"},
                ),
            ]
        )

        conversion = MagicMock()
        conversion.load_dataset = AsyncMock(
            return_value=MagicMock(
                image_count=100,
                annotation_count=500,
            )
        )
        conversion.convert_dataset = AsyncMock(
            return_value=MagicMock(
                id="canonical_001",
                name="test",
                image_count=100,
                annotation_count=500,
                class_count=10,
            )
        )
        conversion.export_dataset = AsyncMock(
            return_value=MagicMock(
                export_format="yolo",
                output_path="/out",
                images_exported=100,
                annotations_exported=500,
                file_count=10,
                file_size_bytes=1024000,
                model_dump=lambda: {"export_format": "yolo"},
            )
        )

        quality = MagicMock()
        quality.run_pipeline = AsyncMock(
            return_value=MagicMock(
                dataset_name="test",
                overall_score=MagicMock(
                    overall=85.5,
                    letter_grade=MagicMock(value="B"),
                    production_ready=True,
                ),
                error_count=2,
                warning_count=5,
            )
        )

        training = MagicMock()
        training.run_pipeline.return_value = MagicMock(
            experiment_id="exp_001",
            model_id="model_001",
            total_epochs_completed=50,
            best_metric=0.85,
            status="completed",
        )

        return DatasetWorkflow(
            catalog_service=svc,
            ontology_service=ontology,
            conversion_service=conversion,
            quality_service=quality,
            training_service=training,
        )

    return DatasetWorkflow(catalog_service=svc)


# ------------------------------------------------------------------
# Fresh ingestion (no duplicate)
# ------------------------------------------------------------------


class TestFirstIngestion:
    @pytest.fixture
    def context(self) -> DatasetContext:
        return DatasetContext(
            dataset_name="test_dataset",
            source_path=Path("/data/test_dataset"),
            dataset_type="coco_json",
        )

    @pytest.mark.asyncio
    async def test_discover_succeeds_first_time(self, context: DatasetContext) -> None:
        """First call to discover() returns a new entry."""
        catalog = MagicMock()
        catalog.discover.return_value = MagicMock(
            entry_id="entry_001",
            name="test_dataset",
            source_type="filesystem",
            model_dump=lambda: {"entry_id": "entry_001"},
        )
        wf = _make_workflow(catalog.discover)
        stage = await wf._run_catalog_stage(context, dry_run=False)
        assert stage.status == PipelineStatus.COMPLETED
        assert stage.result is not None
        assert stage.result.get("entry_id") == "entry_001"

    @pytest.mark.asyncio
    async def test_full_pipeline_first_run(
        self,
        context: DatasetContext,
    ) -> None:
        """Full pipeline completes on first run with normal discover."""
        catalog = MagicMock()
        catalog.discover.return_value = MagicMock(
            entry_id="entry_001",
            name="test_dataset",
            source_type="filesystem",
            model_dump=lambda: {"entry_id": "entry_001"},
        )
        wf = _make_workflow(catalog.discover)
        result = await wf.execute(context)
        assert result.status == PipelineStatus.COMPLETED
        assert result.context is not None
        assert result.context.catalog_entry_id == "entry_001"


# ------------------------------------------------------------------
# Repeated ingestion (CatalogError raised)
# ------------------------------------------------------------------


class TestRepeatedIngestion:
    @pytest.fixture
    def context(self) -> DatasetContext:
        return DatasetContext(
            dataset_name="test_dataset",
            source_path=Path("/data/test_dataset"),
            dataset_type="coco_json",
        )

    @pytest.mark.asyncio
    async def test_reuse_entry_on_catalog_error(self, context: DatasetContext) -> None:
        """When discover raises CatalogError, the entry is reused."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("pipeline_test")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        stage = await wf._run_catalog_stage(context, dry_run=False)
        assert stage.status == PipelineStatus.COMPLETED
        assert stage.result is not None
        assert stage.result.get("entry_id") == "pipeline_test"
        assert stage.result.get("reused") is True

    @pytest.mark.asyncio
    async def test_reuse_sets_context_catalog_entry_id(self, context: DatasetContext) -> None:
        """Context catalog_entry_id is set on reuse."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("reused_entry_42")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        await wf._run_catalog_stage(context, dry_run=False)
        assert context.catalog_entry_id == "reused_entry_42"

    @pytest.mark.asyncio
    async def test_reuse_marks_metadata(self, context: DatasetContext) -> None:
        """Context metadata indicates catalog entry was reused."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("pipeline_test")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        await wf._run_catalog_stage(context, dry_run=False)
        assert context.metadata.get("catalog_entry_reused") is True
        assert context.metadata.get("catalog_entry") is not None
        entry = context.metadata["catalog_entry"]
        assert entry.get("entry_id") == "pipeline_test"
        assert entry.get("name") == "test_dataset"

    @pytest.mark.asyncio
    async def test_reuse_sets_catalog_source_type(self, context: DatasetContext) -> None:
        """Catalog source type is set on reuse."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("pipeline_test")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        await wf._run_catalog_stage(context, dry_run=False)
        assert context.metadata.get("catalog_source_type") is not None

    @pytest.mark.asyncio
    async def test_parse_entry_id_from_error_message(self, context: DatasetContext) -> None:
        """Entry ID is correctly parsed from various error formats."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("my_custom_entry")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        await wf._run_catalog_stage(context, dry_run=False)
        assert context.catalog_entry_id == "my_custom_entry"

    @pytest.mark.asyncio
    async def test_full_pipeline_with_reuse(self, context: DatasetContext) -> None:
        """Full pipeline completes even when catalog entry already exists."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("pipeline_test")
        wf = _make_workflow(catalog.discover)
        result = await wf.execute(context)
        assert result.status == PipelineStatus.COMPLETED
        assert result.context is not None
        assert result.context.catalog_entry_id == "pipeline_test"

    @pytest.mark.asyncio
    async def test_downstream_stages_work_after_reuse(
        self,
        context: DatasetContext,
    ) -> None:
        """Mapping, conversion, and quality stages still work after catalog reuse."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("pipeline_test")
        wf = _make_workflow(catalog.discover)
        result = await wf.execute(context)
        assert result.status == PipelineStatus.COMPLETED
        assert result.context is not None
        assert result.context.canonical_dataset is not None
        assert result.context.quality_report is not None
        stage_names = [s.stage for s in result.stages]
        assert stage_names == ["catalog", "mapping", "conversion", "quality", "training"]
        for s in result.stages:
            assert s.status == PipelineStatus.COMPLETED, f"Stage {s.stage} failed: {s.error}"

    @pytest.mark.asyncio
    async def test_catalog_stage_result_marks_reused(
        self,
        context: DatasetContext,
    ) -> None:
        """Catalog stage result indicates reuse."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("pipeline_test")
        wf = _make_workflow(catalog.discover)
        result = await wf.execute(context)
        catalog_stage = next(s for s in result.stages if s.stage == "catalog")
        assert catalog_stage.result is not None
        assert catalog_stage.result.get("reused") is True


# ------------------------------------------------------------------
# Force flag
# ------------------------------------------------------------------


class TestForceFlag:
    @pytest.fixture
    def context(self) -> DatasetContext:
        return DatasetContext(
            dataset_name="force_test",
            source_path=Path("/data/force_test"),
            dataset_type="coco_json",
        )

    @pytest.mark.asyncio
    async def test_force_with_existing_entry(self, context: DatasetContext) -> None:
        """With --force, existing entry is still reused (no delete API)."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("force_entry")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        stage = await wf._run_catalog_stage(context, dry_run=False, force=True)
        assert stage.status == PipelineStatus.COMPLETED
        assert stage.result is not None
        assert stage.result.get("reused") is True

    @pytest.mark.asyncio
    async def test_force_full_pipeline(self, context: DatasetContext) -> None:
        """Full pipeline with --force and existing entry completes."""
        catalog = MagicMock()
        catalog.discover.side_effect = _make_catalog_error("force_entry")
        wf = _make_workflow(catalog.discover)
        result = await wf.execute(context, force=True)
        assert result.status == PipelineStatus.COMPLETED
        assert result.context is not None
        assert result.context.catalog_entry_id == "force_entry"

    @pytest.mark.asyncio
    async def test_force_first_run(self, context: DatasetContext) -> None:
        """--force on first run still creates entry normally."""
        catalog = MagicMock()
        catalog.discover.return_value = MagicMock(
            entry_id="fresh_entry",
            name="force_test",
            source_type="filesystem",
            model_dump=lambda: {"entry_id": "fresh_entry"},
        )
        wf = _make_workflow(catalog.discover)
        result = await wf.execute(context, force=True)
        assert result.status == PipelineStatus.COMPLETED
        assert result.context is not None
        assert result.context.catalog_entry_id == "fresh_entry"


# ------------------------------------------------------------------
# Non-catalog errors still fail
# ------------------------------------------------------------------


class TestOtherErrorsStillFail:
    @pytest.fixture
    def context(self) -> DatasetContext:
        return DatasetContext(
            dataset_name="error_test",
            source_path=Path("/data/error_test"),
            dataset_type="coco_json",
        )

    @pytest.mark.asyncio
    async def test_non_catalog_error_still_fails(self, context: DatasetContext) -> None:
        """Non-CatalogError exceptions still cause stage failure."""
        catalog = MagicMock()
        catalog.discover.side_effect = RuntimeError("Disk full")
        wf = _make_workflow(catalog.discover, with_full_mocks=False)
        stage = await wf._run_catalog_stage(context, dry_run=False)
        assert stage.status == PipelineStatus.FAILED
        assert stage.error is not None
        assert "Disk full" in stage.error

    @pytest.mark.asyncio
    async def test_dry_run_skipped(self, context: DatasetContext) -> None:
        """Dry run skips catalog regardless of duplicate."""
        wf = _make_workflow(MagicMock(), with_full_mocks=False)
        stage = await wf._run_catalog_stage(context, dry_run=True)
        assert stage.status == PipelineStatus.SKIPPED
        assert stage.details.get("dry_run") is True
