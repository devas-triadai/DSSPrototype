"""Tests for pipeline data models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from backend.dataset_pipeline.models import (
    DatasetContext,
    ExecutionSummary,
    PipelineResult,
    PipelineStageResult,
    PipelineStatus,
)


class TestPipelineStatus:
    def test_values(self) -> None:
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.SKIPPED.value == "skipped"

    def test_all_members_have_unique_values(self) -> None:
        values = [s.value for s in PipelineStatus]
        assert len(values) == len(set(values))


class TestPipelineStageResult:
    def test_minimal_creation(self) -> None:
        r = PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED)
        assert r.stage == "catalog"
        assert r.status == PipelineStatus.COMPLETED
        assert r.result is None
        assert r.error is None
        assert r.duration_seconds == 0.0
        assert r.details == {}

    def test_with_result_and_error(self) -> None:
        r = PipelineStageResult(
            stage="training",
            status=PipelineStatus.FAILED,
            result={"epochs": 10},
            error="OOM error",
            duration_seconds=5.5,
            details={"gpu": "A100"},
        )
        assert r.stage == "training"
        assert r.status == PipelineStatus.FAILED
        assert r.result == {"epochs": 10}
        assert r.error == "OOM error"
        assert r.duration_seconds == 5.5
        assert r.details == {"gpu": "A100"}

    def test_immutable(self) -> None:
        r = PipelineStageResult(stage="test", status=PipelineStatus.PENDING)
        with pytest.raises((TypeError, ValidationError)):
            r.stage = "changed"

    def test_repr(self) -> None:
        r = PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED)
        assert "catalog" in repr(r)
        assert "completed" in repr(r)


class TestExecutionSummary:
    def test_default_creation(self) -> None:
        s = ExecutionSummary()
        assert s.total_duration_seconds == 0.0
        assert s.stages_completed == 0
        assert s.stages_failed == 0
        assert s.stages_skipped == 0
        assert s.stages_total == 0

    def test_custom_values(self) -> None:
        s = ExecutionSummary(
            total_duration_seconds=120.5,
            stages_completed=4,
            stages_failed=1,
            stages_skipped=2,
            stages_total=7,
        )
        assert s.total_duration_seconds == 120.5
        assert s.stages_completed == 4
        assert s.stages_failed == 1

    def test_immutable(self) -> None:
        s = ExecutionSummary()
        with pytest.raises((TypeError, ValidationError)):
            s.stages_completed = 5


class TestDatasetContext:
    def test_minimal_creation(self) -> None:
        ctx = DatasetContext(
            dataset_name="test",
            source_path=Path("/data/test"),
        )
        assert ctx.dataset_name == "test"
        assert ctx.source_path == Path("/data/test")
        assert ctx.dataset_type == ""
        assert ctx.catalog_entry_id is None
        assert ctx.metadata == {}

    def test_full_creation(self) -> None:
        ctx = DatasetContext(
            dataset_name="coco2017",
            source_path=Path("/data/coco"),
            dataset_type="coco_json",
            catalog_entry_id="entry_001",
            metadata={"key": "value"},
        )
        assert ctx.catalog_entry_id == "entry_001"
        assert ctx.metadata == {"key": "value"}

    def test_with_canonical_dataset(self) -> None:
        ctx = DatasetContext(
            dataset_name="test",
            source_path=Path("/data/test"),
            canonical_dataset={"id": "canonical_001"},
        )
        assert ctx.canonical_dataset == {"id": "canonical_001"}


class TestPipelineResult:
    def test_minimal_creation(self) -> None:
        r = PipelineResult(
            status=PipelineStatus.PENDING,
            dataset_name="test",
        )
        assert r.status == PipelineStatus.PENDING
        assert r.dataset_name == "test"
        assert r.stages == []
        assert r.error is None

    def test_with_stages_and_context(self, sample_dataset_context: Any) -> None:
        stages = [
            PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED),
            PipelineStageResult(stage="mapping", status=PipelineStatus.COMPLETED),
        ]
        r = PipelineResult(
            status=PipelineStatus.COMPLETED,
            dataset_name="test",
            stages=stages,
            context=sample_dataset_context,
        )
        assert len(r.stages) == 2
        assert r.context is not None
        assert r.context.dataset_name == "test_dataset"

    def test_with_error(self) -> None:
        r = PipelineResult(
            status=PipelineStatus.FAILED,
            dataset_name="test",
            error="Pipeline failed",
        )
        assert r.error == "Pipeline failed"

    def test_stages_mutable(self) -> None:
        r = PipelineResult(
            status=PipelineStatus.RUNNING,
            dataset_name="test",
        )
        r.stages.append(PipelineStageResult(stage="catalog", status=PipelineStatus.COMPLETED))
        assert len(r.stages) == 1
