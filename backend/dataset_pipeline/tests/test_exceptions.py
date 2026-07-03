"""Tests for pipeline exceptions."""

from __future__ import annotations

import pytest

from backend.dataset_pipeline.exceptions import (
    DatasetNotFoundError,
    PipelineError,
    PipelineValidationError,
    StageExecutionError,
)


class TestPipelineError:
    def test_base_exception(self) -> None:
        exc = PipelineError("base error")
        assert isinstance(exc, Exception)
        assert str(exc) == "base error"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(PipelineError):
            raise PipelineError("test")


class TestStageExecutionError:
    def test_creation_with_stage_and_message(self) -> None:
        exc = StageExecutionError("catalog", "Catalog failed")
        assert exc.stage == "catalog"
        assert "catalog" in str(exc)
        assert "Catalog failed" in str(exc)

    def test_with_cause(self) -> None:
        cause = ValueError("underlying issue")
        exc = StageExecutionError("training", "Training failed", cause=cause)
        assert exc.cause is cause
        assert str(cause) in str(exc)

    def test_error_string_format(self) -> None:
        exc = StageExecutionError("mapping", "Invalid label")
        expected = "[mapping] Invalid label"
        assert str(exc) == expected

    def test_subclass_of_pipeline_error(self) -> None:
        assert issubclass(StageExecutionError, PipelineError)


class TestDatasetNotFoundError:
    def test_with_name_only(self) -> None:
        exc = DatasetNotFoundError("coco2017")
        assert exc.dataset_name == "coco2017"
        assert exc.path is None
        assert "coco2017" in str(exc)

    def test_with_name_and_path(self) -> None:
        exc = DatasetNotFoundError("coco2017", path="/data/coco2017")
        assert exc.path == "/data/coco2017"
        assert "/data/coco2017" in str(exc)

    def test_subclass_of_pipeline_error(self) -> None:
        assert issubclass(DatasetNotFoundError, PipelineError)


class TestPipelineValidationError:
    def test_with_message_only(self) -> None:
        exc = PipelineValidationError("invalid input")
        assert exc.errors == []
        assert str(exc) == "invalid input"

    def test_with_message_and_errors(self) -> None:
        errors = ["dataset name required", "source path must exist"]
        exc = PipelineValidationError("validation failed", errors=errors)
        assert exc.errors == errors

    def test_subclass_of_pipeline_error(self) -> None:
        assert issubclass(PipelineValidationError, PipelineError)
