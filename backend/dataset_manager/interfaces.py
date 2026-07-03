"""Abstract interfaces for every component in the Dataset Management Platform.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_manager.models import (
    DatasetChecksum,
    DatasetExport,
    DatasetInfo,
    DatasetMetadata,
    DatasetQuality,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidation,
    DatasetVersion,
)


class RegistryInterface(ABC):
    """Contract for the dataset registry."""

    @abstractmethod
    def register(self, info: DatasetInfo) -> DatasetInfo:
        """Register a new dataset in the registry.

        Returns the registered DatasetInfo with registry-assigned fields.
        """

    @abstractmethod
    def get(self, dataset_id: str) -> DatasetInfo | None:
        """Retrieve dataset metadata by ID."""

    @abstractmethod
    def get_by_name(self, name: str) -> DatasetInfo | None:
        """Retrieve dataset metadata by name."""

    @abstractmethod
    def list_datasets(self) -> list[DatasetInfo]:
        """Return all registered datasets."""

    @abstractmethod
    def update(self, info: DatasetInfo) -> DatasetInfo:
        """Update an existing dataset's metadata."""

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Remove a dataset from the registry.

        Returns True if the dataset existed and was removed.
        """

    @abstractmethod
    def contains(self, dataset_id: str) -> bool:
        """Check whether a dataset exists in the registry."""


class DatasetLoaderInterface(ABC):
    """Contract for loading datasets from disk."""

    @abstractmethod
    def load_images(self, path: Path) -> list[Path]:
        """Return all image file paths under *path*.

        Supports recursive discovery through subdirectories.
        """

    @abstractmethod
    def load_annotations(self, path: Path) -> list[Path]:
        """Return all annotation file paths under *path*."""


class ValidationEngineInterface(ABC):
    """Contract for dataset validation."""

    @abstractmethod
    def validate(
        self, dataset_path: Path, annotation_path: Path | None = None,
    ) -> DatasetValidation:
        """Run all validation checks on a dataset.

        Returns a DatasetValidation report.
        """


class StatisticsEngineInterface(ABC):
    """Contract for computing dataset statistics."""

    @abstractmethod
    def compute(self, dataset_path: Path, annotation_path: Path | None = None) -> DatasetStatistics:
        """Compute comprehensive statistics for a dataset."""


class QualityEngineInterface(ABC):
    """Contract for assessing dataset quality."""

    @abstractmethod
    def assess(
        self,
        statistics: DatasetStatistics,
        validation: DatasetValidation,
    ) -> DatasetQuality:
        """Compute a quality score and report for a dataset."""


class VersioningInterface(ABC):
    """Contract for dataset version management."""

    @abstractmethod
    def create_version(
        self,
        dataset_id: str,
        version: str | None = None,
        change_log: str = "",
    ) -> DatasetVersion:
        """Create a new version for a dataset.

        If *version* is None, auto-increment from the latest version.
        Returns the created DatasetVersion.
        """

    @abstractmethod
    def get_version(self, dataset_id: str, version: str) -> DatasetVersion | None:
        """Retrieve a specific version of a dataset."""

    @abstractmethod
    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        """Return all versions for a dataset, ordered newest first."""

    @abstractmethod
    def get_latest_version(self, dataset_id: str) -> DatasetVersion | None:
        """Return the most recent version of a dataset."""


class SplitterInterface(ABC):
    """Contract for dataset splitting."""

    @abstractmethod
    def split(
        self,
        images: Sequence[Path],
        annotations: Sequence[Path] | None = None,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = False,
    ) -> DatasetSplit:
        """Split images (and optional annotations) into train/validation/test sets.

        Returns a DatasetSplit with absolute paths.
        """


class DatasetExporterInterface(ABC):
    """Contract for exporting datasets to a specific format."""

    @abstractmethod
    def export(
        self,
        images: Sequence[Path],
        annotations: Sequence[Path],
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
    ) -> DatasetExport:
        """Export a dataset to the target format.

        Returns an DatasetExport containing metadata about the export.
        """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of the export format (e.g. 'yolo', 'coco', 'voc')."""


class MetadataGeneratorInterface(ABC):
    """Contract for generating dataset metadata."""

    @abstractmethod
    def generate(
        self,
        dataset_info: DatasetInfo,
        statistics: DatasetStatistics | None = None,
        quality: DatasetQuality | None = None,
        validation: DatasetValidation | None = None,
        checksum: DatasetChecksum | None = None,
    ) -> DatasetMetadata:
        """Generate a complete metadata record for a dataset."""


class ChecksumInterface(ABC):
    """Contract for checksum operations."""

    @abstractmethod
    def compute(self, path: Path) -> DatasetChecksum:
        """Compute a checksum for the file or directory at *path*."""

    @abstractmethod
    def verify(self, path: Path, expected: str) -> bool:
        """Verify that *path* matches the expected checksum.

        Returns True if the checksum matches.
        """
