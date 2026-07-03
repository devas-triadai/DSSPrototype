"""Tests for DatasetIngestor."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from backend.dataset_pipeline.exceptions import (
    DatasetNotFoundError,
    PipelineValidationError,
)
from backend.dataset_pipeline.ingest import DatasetIngestor
from backend.dataset_pipeline.models import (
    DatasetContext,
    DatasetLayout,
    PipelineResult,
    PipelineStatus,
)


class TestDatasetIngestor:
    @pytest.fixture
    def mock_workflow(self) -> MagicMock:
        wf = MagicMock()

        async def _execute(context: DatasetContext, **kw: Any) -> PipelineResult:
            return PipelineResult(
                status=PipelineStatus.COMPLETED,
                dataset_name=context.dataset_name,
                stages=[],
            )

        wf.execute = _execute
        return wf

    @pytest.fixture
    def ingestor(self, mock_workflow: MagicMock) -> DatasetIngestor:
        return DatasetIngestor(workflow=mock_workflow)

    # ------------------------------------------------------------------
    # Successful ingestion
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_completes(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_with_skip_quality(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            skip_quality=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_with_skip_training(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            skip_training=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_with_continue_on_error(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            continue_on_error=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_ingest_with_output_dir(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
        tmp_path: Path,
    ) -> None:
        output_dir = tmp_path / "output"
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            output_dir=output_dir,
        )
        assert result.status == PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # Dry run
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_dry_run(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
            dry_run=True,
        )
        assert result.status == PipelineStatus.COMPLETED

    # ------------------------------------------------------------------
    # Source validation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_validate_source_with_directory(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        layout = await ingestor.validate_source(temp_source_dir)
        assert isinstance(layout, DatasetLayout)
        assert layout.dataset_type == "unknown"
        assert temp_source_dir.resolve() in layout.image_directories

    @pytest.mark.asyncio
    async def test_validate_source_with_json_file(
        self,
        ingestor: DatasetIngestor,
        tmp_path: Path,
    ) -> None:
        json_file = tmp_path / "coco_annotations.json"
        json_file.write_text('{"images": []}')
        layout = await ingestor.validate_source(json_file)
        assert isinstance(layout, DatasetLayout)
        assert json_file.resolve() in layout.annotation_files

    @pytest.mark.asyncio
    async def test_validate_source_with_yaml_file(
        self,
        ingestor: DatasetIngestor,
        tmp_path: Path,
    ) -> None:
        yaml_file = tmp_path / "data.yaml"
        yaml_file.write_text("train: ./images/train")
        layout = await ingestor.validate_source(yaml_file)
        assert isinstance(layout, DatasetLayout)
        assert yaml_file.resolve() in layout.annotation_files

    @pytest.mark.asyncio
    async def test_validate_source_accepts_any_directory_with_images(
        self,
        ingestor: DatasetIngestor,
        tmp_path: Path,
    ) -> None:
        src = tmp_path / "my_dataset"
        src.mkdir()
        (src / "img.jpg").write_text("data")
        layout = await ingestor.validate_source(src)
        assert isinstance(layout, DatasetLayout)
        assert src.resolve() in layout.image_directories

    # ------------------------------------------------------------------
    # Error cases
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingest_nonexistent_path(self, ingestor: DatasetIngestor) -> None:
        with pytest.raises(DatasetNotFoundError):
            await ingestor.ingest(
                dataset_name="missing",
                source_path=Path("/nonexistent/path"),
            )

    @pytest.mark.asyncio
    async def test_validate_nonexistent_source(
        self,
        ingestor: DatasetIngestor,
    ) -> None:
        with pytest.raises(DatasetNotFoundError):
            await ingestor.validate_source(Path("/does/not/exist"))

    @pytest.mark.asyncio
    async def test_validate_source_empty_directory(
        self,
        ingestor: DatasetIngestor,
        tmp_path: Path,
    ) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(PipelineValidationError):
            await ingestor.validate_source(empty_dir)

    @pytest.mark.asyncio
    async def test_validate_source_unsupported_file_type(
        self,
        ingestor: DatasetIngestor,
        tmp_path: Path,
    ) -> None:
        bad_file = tmp_path / "data.zip"
        bad_file.write_text("nope")
        with pytest.raises(PipelineValidationError):
            await ingestor.validate_source(bad_file)

    # ------------------------------------------------------------------
    # Error propagation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ingestor_returns_pipeline_result(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="test_dataset",
            source_path=temp_source_dir,
        )
        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_ingestor_sets_dataset_name(
        self,
        ingestor: DatasetIngestor,
        temp_source_dir: Path,
    ) -> None:
        result = await ingestor.ingest(
            dataset_name="custom_name",
            source_path=temp_source_dir,
        )
        assert result.dataset_name == "custom_name"
