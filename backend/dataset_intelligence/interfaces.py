"""Abstract interfaces for every component in the Dataset Intelligence Pipeline.

All concrete implementations depend on these contracts, never on each other.
The pipeline is model-agnostic and format-agnostic.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_intelligence.models import (
    DatasetIntelligenceRegistryEntry,
    DuplicateReport,
    ExportResult,
    HarmonizedDataset,
    MergedDataset,
    NormalizedDataset,
    OntologyMappingReport,
    ProcessedDataset,
    QualityReport,
    RawDataset,
    StatisticsReport,
    ValidationReport,
)


class FormatParserInterface(ABC):
    """Contract for parsing datasets from raw import formats."""

    @abstractmethod
    def parse(self, source_path: Path) -> RawDataset:
        """Parse a dataset from disk into the canonical RawDataset model.

        Parameters
        ----------
        source_path:
            Path to the dataset directory or file.

        Returns
        -------
        RawDataset
            The parsed dataset with images, annotations, and metadata.
        """

    @property
    @abstractmethod
    def supported_formats(self) -> list[str]:
        """Return the list of format identifiers this parser supports."""


class FormatParserRegistryInterface(ABC):
    """Contract for registering and retrieving format parsers."""

    @abstractmethod
    def register(self, parser: FormatParserInterface) -> None:
        """Register a format parser."""

    @abstractmethod
    def get_parser(self, format_name: str) -> FormatParserInterface:
        """Retrieve a parser by format name."""

    @abstractmethod
    def detect_format(self, source_path: Path) -> str:
        """Auto-detect the format of a dataset at *source_path*."""


class DatasetValidatorInterface(ABC):
    """Contract for dataset validation."""

    @abstractmethod
    def validate(self, dataset: RawDataset) -> ValidationReport:
        """Validate a raw dataset and return a report.

        Checks:
          - Image/annotation consistency
          - Bounding box validity
          - Class consistency
          - File integrity
        """


class DatasetNormalizerInterface(ABC):
    """Contract for dataset normalization."""

    @abstractmethod
    def normalize(self, dataset: RawDataset) -> NormalizedDataset:
        """Normalize a raw dataset into a canonical representation.

        Normalizes:
          - Class names (lowercase, underscores)
          - File naming conventions
          - Bounding box formats (to xyxy_normalized)
          - Directory structure
        """


class OntologyMapperInterface(ABC):
    """Contract for mapping dataset classes to the DSS ontology."""

    @abstractmethod
    def map_classes(self, dataset: NormalizedDataset) -> OntologyMappingReport:
        """Map all classes in *dataset* to canonical ontology concepts.

        Returns an OntologyMappingReport with mappings, unmapped classes,
        and ontology coverage statistics.
        """

    @abstractmethod
    def apply_mapping(
        self, dataset: NormalizedDataset, report: OntologyMappingReport
    ) -> NormalizedDataset:
        """Apply an ontology mapping to a dataset, returning an updated copy."""


class DuplicateDetectorInterface(ABC):
    """Contract for duplicate detection in datasets."""

    @abstractmethod
    def detect(self, dataset: NormalizedDataset) -> DuplicateReport:
        """Detect duplicates in a dataset.

        Detects:
          - Identical filenames
          - Identical hashes
          - Identical metadata
          - Near-duplicate images (architecture-ready)
          - Duplicate annotations
        """


class ClassHarmonizerInterface(ABC):
    """Contract for harmonizing class names across datasets."""

    @abstractmethod
    def harmonize(
        self, dataset: NormalizedDataset, ontology_mapping: OntologyMappingReport
    ) -> HarmonizedDataset:
        """Harmonize class names using ontology mappings.

        Example: ``tank``, ``MBT``, ``battle tank`` → ``main_battle_tank``
        """

    @abstractmethod
    def build_harmonization_mapping(
        self, classes: Sequence[str], ontology_mapping: OntologyMappingReport
    ) -> dict[str, str]:
        """Build a mapping from raw class names to harmonized names."""


class DatasetMergerInterface(ABC):
    """Contract for merging multiple datasets into one."""

    @abstractmethod
    def merge(self, datasets: Sequence[NormalizedDataset | HarmonizedDataset]) -> MergedDataset:
        """Merge multiple datasets into a unified dataset.

        Maintains provenance for every sample.
        """


class DatasetSplitterInterface(ABC):
    """Contract for splitting a dataset into train/validation/test."""

    @abstractmethod
    def split(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = True,
    ) -> dict[str, list[str]]:
        """Split a dataset into train/validation/test sets.

        Returns a dict mapping split name to lists of image IDs.
        """


class StatisticsEngineInterface(ABC):
    """Contract for computing dataset statistics."""

    @abstractmethod
    def compute(
        self, dataset: NormalizedDataset | HarmonizedDataset | MergedDataset
    ) -> StatisticsReport:
        """Compute comprehensive statistics for a dataset."""


class QualityEngineInterface(ABC):
    """Contract for assessing dataset quality."""

    @abstractmethod
    def assess(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        validation: ValidationReport | None,
        duplicates: DuplicateReport | None,
        statistics: StatisticsReport | None,
    ) -> QualityReport:
        """Assess dataset quality and return a report."""


class DatasetExporterInterface(ABC):
    """Contract for exporting datasets to training formats."""

    @abstractmethod
    def export(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        output_dir: Path,
        class_mapping: dict[str, int] | None = None,
        splits: dict[str, list[str]] | None = None,
    ) -> ExportResult:
        """Export a dataset to the target format.

        Returns metadata about the export.
        """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """Return the name of the export format (e.g. 'yolo', 'coco', 'voc')."""


class DatasetIntelligenceRegistryInterface(ABC):
    """Contract for the dataset intelligence registry."""

    @abstractmethod
    def register(
        self, entry: "DatasetIntelligenceRegistryEntry"
    ) -> "DatasetIntelligenceRegistryEntry":
        """Register a processed dataset."""

    @abstractmethod
    def get(self, dataset_id: str) -> "DatasetIntelligenceRegistryEntry | None":
        """Retrieve a registry entry by dataset ID."""

    @abstractmethod
    def list_entries(self) -> list["DatasetIntelligenceRegistryEntry"]:
        """List all registry entries."""

    @abstractmethod
    def update(
        self, entry: "DatasetIntelligenceRegistryEntry"
    ) -> "DatasetIntelligenceRegistryEntry":
        """Update a registry entry."""

    @abstractmethod
    def delete(self, dataset_id: str) -> bool:
        """Delete a registry entry."""


class DatasetIntelligenceServiceInterface(ABC):
    """Contract for the public Dataset Intelligence service."""

    @abstractmethod
    def import_dataset(
        self,
        source_path: Path,
        dataset_name: str,
        format_hint: str | None = None,
    ) -> ProcessedDataset:
        """Import and process a dataset through the full pipeline.

        Returns a ProcessedDataset ready for training or registration.
        """

    @abstractmethod
    def merge_datasets(
        self,
        dataset_ids: Sequence[str],
        merged_name: str,
    ) -> ProcessedDataset:
        """Merge multiple processed datasets into one unified dataset."""

    @abstractmethod
    def export_dataset(
        self,
        dataset_id: str,
        format_name: str,
        output_dir: Path | None = None,
    ) -> ExportResult:
        """Export a processed dataset to a specific format."""

    @abstractmethod
    def get_quality_report(self, dataset_id: str) -> QualityReport | None:
        """Retrieve the quality report for a processed dataset."""

    @abstractmethod
    def get_statistics(self, dataset_id: str) -> StatisticsReport | None:
        """Retrieve the statistics for a processed dataset."""
