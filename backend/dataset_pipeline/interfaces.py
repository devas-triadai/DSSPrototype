"""Abstract interfaces for the dataset pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from backend.dataset_pipeline.models import DatasetContext, DatasetLayout, PipelineResult


class WorkflowInterface(ABC):
    """Defines the contract for the dataset workflow orchestrator."""

    @abstractmethod
    async def execute(
        self,
        context: DatasetContext,
        skip_quality: bool = False,
        skip_training: bool = False,
        continue_on_error: bool = False,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult: ...


class PipelineServiceInterface(ABC):
    """Public facade for the dataset pipeline."""

    @abstractmethod
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
    ) -> PipelineResult: ...

    @abstractmethod
    async def run_catalog(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult: ...

    @abstractmethod
    async def run_mapping(
        self,
        dataset_name: str,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult: ...

    @abstractmethod
    async def run_conversion(
        self,
        dataset_name: str,
        source_path: Path,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult: ...

    @abstractmethod
    async def run_quality(
        self,
        dataset_name: str,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult: ...

    @abstractmethod
    async def run_training(
        self,
        dataset_name: str,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> PipelineResult: ...

    @abstractmethod
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
    ) -> PipelineResult: ...


class DatasetIngestorInterface(ABC):
    """Defines the contract for dataset ingestion."""

    @abstractmethod
    async def ingest(
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
    ) -> PipelineResult: ...

    @abstractmethod
    async def validate_source(self, source_path: Path) -> DatasetLayout:
        """Validate the dataset source path and return a structured layout."""
        ...
