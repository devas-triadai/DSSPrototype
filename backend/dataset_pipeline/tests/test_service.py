"""Tests for PipelineService facade."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from backend.dataset_pipeline.models import PipelineStatus
from backend.dataset_pipeline.service import PipelineService


class TestPipelineService:
    @pytest.fixture
    def service(
        self,
        mock_workflow: MagicMock,
        mock_catalog_service: MagicMock,
    ) -> PipelineService:
        ingestor = MagicMock()
        async def _ingest(dataset_name: str = "", **kw: object) -> MagicMock:
            return MagicMock(
                status=PipelineStatus.COMPLETED,
                dataset_name=dataset_name,
                stages=[],
                summary=MagicMock(
                    stages_completed=0, stages_failed=0, stages_skipped=0, stages_total=0,
                ),
                error=None,
                context=None,
            )
        ingestor.ingest = _ingest
        return PipelineService(ingestor=ingestor, workflow=mock_workflow)

    # ------------------------------------------------------------------
    # ingest_dataset
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_dataset(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_dataset_with_skip_training(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            skip_training=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_dataset_with_skip_quality(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            skip_quality=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_dataset_dry_run(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            dry_run=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_dataset_continue_on_error(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            continue_on_error=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # Individual stage methods
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_catalog(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.run_catalog(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_catalog_dry_run(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.run_catalog(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            dry_run=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_mapping(self, service: PipelineService) -> None:
        result = await service.run_mapping(dataset_name="test_dataset")
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_conversion(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.run_conversion(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_quality(self, service: PipelineService) -> None:
        result = await service.run_quality(dataset_name="test_dataset")
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_training(self, service: PipelineService) -> None:
        result = await service.run_training(dataset_name="test_dataset")
        assert result.status == PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # run_full_pipeline
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_run_full_pipeline(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.run_full_pipeline(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_full_pipeline_with_skips(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.run_full_pipeline(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            skip_quality=True,
            skip_training=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_run_full_pipeline_dry_run(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.run_full_pipeline(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            dry_run=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # Return type checks
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_result_has_dataset_name(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="custom_name",
            source_path=temp_source_dir,
        )
        assert result.dataset_name == "custom_name"

    @pytest.mark.asyncio
    async def test_result_has_summary(
        self, service: PipelineService, temp_source_dir: Path,
    ) -> None:
        result = await service.ingest_dataset(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert result.summary is not None
