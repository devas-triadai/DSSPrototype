"""Shared fixtures and mocks for dataset_pipeline tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.dataset_pipeline.config import PipelineConfig
from backend.dataset_pipeline.models import (
    DatasetContext,
    PipelineResult,
    PipelineStageResult,
    PipelineStatus,
)

# ------------------------------------------------------------------
# Sample models
# ------------------------------------------------------------------

@pytest.fixture
def sample_dataset_context() -> DatasetContext:
    return DatasetContext(
        dataset_name="test_dataset",
        source_path=Path("/data/test_dataset"),
        dataset_type="coco_json",
    )


@pytest.fixture
def sample_completed_stage() -> PipelineStageResult:
    return PipelineStageResult(
        stage="catalog",
        status=PipelineStatus.COMPLETED,
        result={"entry_id": "test_entry"},
        duration_seconds=1.5,
    )


@pytest.fixture
def sample_failed_stage() -> PipelineStageResult:
    return PipelineStageResult(
        stage="catalog",
        status=PipelineStatus.FAILED,
        error="Something went wrong",
        duration_seconds=0.5,
    )


@pytest.fixture
def sample_pipeline_result_completed(sample_dataset_context: DatasetContext) -> PipelineResult:
    return PipelineResult(
        status=PipelineStatus.COMPLETED,
        dataset_name="test_dataset",
        stages=[
            PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="mapping", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="conversion", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="quality", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="training", status=PipelineStatus.COMPLETED),
        ],
        context=sample_dataset_context,
    )


@pytest.fixture
def sample_pipeline_result_failed(sample_dataset_context: DatasetContext) -> PipelineResult:
    return PipelineResult(
        status=PipelineStatus.FAILED,
        dataset_name="test_dataset",
        stages=[
            PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED),
            PipelineStageResult(
                stage="mapping", status=PipelineStatus.FAILED, error="Mapping error",
            ),
        ],
        context=sample_dataset_context,
        error="[mapping] Mapping error",
    )


# ------------------------------------------------------------------
# Mock services
# ------------------------------------------------------------------

@pytest.fixture
def mock_catalog_service() -> MagicMock:
    svc = MagicMock()
    svc.discover.return_value = MagicMock(
        entry_id="test_entry",
        name="test_dataset",
        source_type="coco_json",
        model_dump=lambda: {"entry_id": "test_entry"},
    )
    return svc


@pytest.fixture
def mock_ontology_service() -> MagicMock:
    svc = MagicMock()
    svc.register_dataset = AsyncMock()
    svc.map_dataset = AsyncMock(return_value=[
        MagicMock(
            source_label="car",
            canonical_value="vehicle.car",
            canonical_name="Car",
            confidence=0.95,
            model_dump=lambda: {"source_label": "car"},
        ),
    ])
    return svc


@pytest.fixture
def mock_conversion_service() -> MagicMock:
    svc = MagicMock()
    load_result = MagicMock(
        image_count=100,
        annotation_count=500,
    )
    canonical = MagicMock(
        id="canonical_001",
        name="test_dataset",
        image_count=100,
        annotation_count=500,
        class_count=10,
    )
    svc.load_dataset = AsyncMock(return_value=load_result)
    svc.convert_dataset = AsyncMock(return_value=canonical)
    export_result = MagicMock(
        export_format="yolo",
        output_path="/output/test_dataset",
        images_exported=100,
        annotations_exported=500,
        file_count=10,
        file_size_bytes=1024000,
        model_dump=lambda: {"export_format": "yolo"},
    )
    svc.export_dataset = AsyncMock(return_value=export_result)
    return svc


@pytest.fixture
def mock_quality_service() -> MagicMock:
    svc = MagicMock()
    report = MagicMock(
        dataset_name="test_dataset",
        overall_score=MagicMock(
            overall=85.5,
            letter_grade=MagicMock(value="B"),
            production_ready=True,
        ),
        error_count=2,
        warning_count=5,
    )
    svc.run_pipeline = AsyncMock(return_value=report)
    return svc


@pytest.fixture
def mock_training_service() -> MagicMock:
    svc = MagicMock()
    result = MagicMock(
        experiment_id="exp_001",
        model_id="model_001",
        total_epochs_completed=50,
        best_metric=0.85,
        status="completed",
    )
    svc.run_pipeline.return_value = result
    return svc


@pytest.fixture
def mock_workflow(mock_catalog_service: MagicMock) -> MagicMock:
    wf = MagicMock()
    wf.execute = AsyncMock(return_value=PipelineResult(
        status=PipelineStatus.COMPLETED,
        dataset_name="test_dataset",
        stages=[
            PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="mapping", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="conversion", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="quality", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="training", status=PipelineStatus.COMPLETED),
        ],
    ))
    completed_stage = PipelineStageResult(
        stage="catalog", status=PipelineStatus.COMPLETED,
        result={"entry_id": "test_entry"}, duration_seconds=0.1,
    )
    wf._run_catalog_stage = AsyncMock(return_value=completed_stage)
    wf._run_mapping_stage = AsyncMock(return_value=PipelineStageResult(
        stage="mapping", status=PipelineStatus.COMPLETED,
        result={"labels_mapped": 1}, duration_seconds=0.1,
    ))
    wf._run_conversion_stage = AsyncMock(return_value=PipelineStageResult(
        stage="conversion", status=PipelineStatus.COMPLETED,
        result={"images": 100, "annotations": 500, "classes": 10},
        duration_seconds=0.1,
    ))
    wf._run_quality_stage = AsyncMock(return_value=PipelineStageResult(
        stage="quality", status=PipelineStatus.COMPLETED,
        result={"quality_score": 85.5, "letter_grade": "B"},
        duration_seconds=0.1,
    ))
    wf._run_training_stage = AsyncMock(return_value=PipelineStageResult(
        stage="training", status=PipelineStatus.COMPLETED,
        result={"experiment_id": "exp_001"}, duration_seconds=0.1,
    ))
    return wf


# ------------------------------------------------------------------
# Temp directories
# ------------------------------------------------------------------

@pytest.fixture
def temp_source_dir(tmp_path: Path) -> Path:
    src = tmp_path / "datasets" / "test_coco"
    src.mkdir(parents=True, exist_ok=True)
    (src / "image_001.jpg").write_text("fake-image-data")
    (src / "image_002.jpg").write_text("fake-image-data")
    (src / "_annotations.json").write_text('{"images": [], "annotations": []}')
    return src


@pytest.fixture
def pipeline_config_override() -> PipelineConfig:
    return PipelineConfig(
        stage_timeout_seconds=10,
        continue_on_error=True,
        dry_run=False,
    )
