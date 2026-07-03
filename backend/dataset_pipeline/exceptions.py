"""Custom exceptions for the dataset pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all dataset pipeline errors."""


class StageExecutionError(PipelineError):
    """Raised when a specific pipeline stage fails."""

    def __init__(self, stage: str, message: str, cause: Exception | None = None) -> None:
        self.stage = stage
        self.cause = cause
        msg = f"[{stage}] {message}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


class DatasetNotFoundError(PipelineError):
    """Raised when the specified dataset cannot be located."""

    def __init__(self, dataset_name: str, path: str | None = None) -> None:
        self.dataset_name = dataset_name
        self.path = path
        msg = f"Dataset '{dataset_name}' not found"
        if path:
            msg += f" at {path}"
        super().__init__(msg)


class PipelineValidationError(PipelineError):
    """Raised when pipeline input validation fails."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        self.errors = errors or []
        super().__init__(message)
