"""DatasetManagementService — public entry point for the Dataset Management Platform.

Coordinates the full dataset lifecycle through dependency injection.
All components are replaceable behind their interfaces.

Responsibilities:
  - load dataset
  - validate dataset
  - generate statistics
  - generate metadata
  - generate checksum
  - register dataset
  - create version
  - split dataset
  - export dataset
"""

import json
import logging
from pathlib import Path

from backend.dataset_manager.checksum import ChecksumGenerator
from backend.dataset_manager.config import dm_config
from backend.dataset_manager.exceptions import (
    DatasetNotFoundError,
    ExportError,
    SplitError,
    ValidationError,
)
from backend.dataset_manager.exporter import CocoExporter, PascalVocExporter, YoloExporter
from backend.dataset_manager.interfaces import (
    ChecksumInterface,
    DatasetExporterInterface,
    DatasetLoaderInterface,
    MetadataGeneratorInterface,
    QualityEngineInterface,
    RegistryInterface,
    SplitterInterface,
    StatisticsEngineInterface,
    ValidationEngineInterface,
    VersioningInterface,
)
from backend.dataset_manager.loader import DatasetLoader
from backend.dataset_manager.metadata import MetadataGenerator
from backend.dataset_manager.models import (
    DatasetExport,
    DatasetInfo,
    DatasetQuality,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidation,
    DatasetVersion,
)
from backend.dataset_manager.quality import QualityEngine
from backend.dataset_manager.registry import DatasetRegistry
from backend.dataset_manager.splitter import DatasetSplitter
from backend.dataset_manager.statistics import StatisticsEngine
from backend.dataset_manager.validator import DatasetValidator
from backend.dataset_manager.versioning import DatasetVersioning

logger = logging.getLogger("dss.dataset_manager.service")


class DatasetManagementService:
    """Coordinates the full dataset lifecycle.

    All dependencies are injected via the constructor with sensible defaults.
    Follows the same pattern as ComputerVisionService in the DSS codebase.
    """

    def __init__(
        self,
        registry: RegistryInterface | None = None,
        loader: DatasetLoaderInterface | None = None,
        validator: ValidationEngineInterface | None = None,
        statistics: StatisticsEngineInterface | None = None,
        quality: QualityEngineInterface | None = None,
        metadata: MetadataGeneratorInterface | None = None,
        checksum: ChecksumInterface | None = None,
        versioning: VersioningInterface | None = None,
        splitter: SplitterInterface | None = None,
        exporters: dict[str, DatasetExporterInterface] | None = None,
    ) -> None:
        self._registry = registry or DatasetRegistry()
        self._loader = loader or DatasetLoader()
        self._validator = validator or DatasetValidator()
        self._statistics = statistics or StatisticsEngine()
        self._quality = quality or QualityEngine()
        self._metadata = metadata or MetadataGenerator()
        self._checksum = checksum or ChecksumGenerator()
        self._versioning = versioning or DatasetVersioning()
        self._splitter = splitter or DatasetSplitter()
        self._exporters = exporters or {
            "yolo": YoloExporter(),
            "coco": CocoExporter(),
            "voc": PascalVocExporter(),
        }

    # ------------------------------------------------------------------
    # Dataset lifecycle
    # ------------------------------------------------------------------

    async def load_and_register(
        self,
        dataset_path: Path,
        dataset_name: str,
        description: str = "",
        source: str = "",
        dataset_type: str = "raw",
    ) -> DatasetInfo:
        """Load a dataset from disk, run full pipeline, and register it."""
        logger.info("Dataset load started: %s (%s)", dataset_name, dataset_path)

        dataset_id = f"{dataset_name}_{dataset_type}"

        info = DatasetInfo(
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            description=description,
            source=source,
        )

        validation = await self.validate_dataset(dataset_path)
        info = DatasetInfo(
            dataset_id=info.dataset_id,
            dataset_name=info.dataset_name,
            dataset_version=info.dataset_version,
            dataset_type=info.dataset_type,
            description=info.description,
            source=info.source,
            license=info.license,
            validation_status="passed" if validation.passed else "failed",
        )

        stats = await self.generate_statistics(dataset_path)

        quality_report = await self.assess_quality(dataset_id)

        checksum = self._checksum.compute(dataset_path)

        self._metadata.generate(
            dataset_info=info,
            statistics=stats,
            quality=quality_report,
            validation=validation,
            checksum=checksum,
        )

        info = DatasetInfo(
            dataset_id=info.dataset_id,
            dataset_name=info.dataset_name,
            dataset_version=info.dataset_version,
            dataset_type=info.dataset_type,
            description=info.description,
            source=info.source,
            license=info.license,
            image_count=stats.total_images,
            annotation_count=stats.total_annotations,
            class_count=stats.class_count,
            classes=stats.classes,
            validation_status="passed" if validation.passed else "failed",
            quality_score=quality_report.quality_score,
            statistics_file=str(dm_config.statistics_dir / f"{dataset_id}_stats.json"),
            metadata_file=str(dm_config.metadata_dir / f"{dataset_id}_metadata.json"),
            checksum=checksum,
        )

        registered = self._registry.register(info)

        version = self._versioning.create_version(
            dataset_id=dataset_id,
            change_log=f"Initial import of {dataset_name}",
        )

        logger.info(
            "Dataset load completed: %s v%s (images=%d, annotations=%d)",
            dataset_id,
            version.version,
            stats.total_images,
            stats.total_annotations,
        )
        return registered

    async def validate_dataset(self, dataset_path: Path) -> DatasetValidation:
        """Validate a dataset and return the report."""
        logger.info("Validation started: %s", dataset_path)
        try:
            return self._validator.validate(dataset_path)
        except Exception as e:
            raise ValidationError(f"Validation failed: {e}") from e

    async def generate_statistics(self, dataset_path: Path) -> DatasetStatistics:
        """Generate statistics for a dataset."""
        logger.info("Statistics generation started: %s", dataset_path)
        return self._statistics.compute(dataset_path)

    async def assess_quality(self, dataset_id: str) -> DatasetQuality:
        """Assess quality for a registered dataset."""
        dataset = self._registry.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")

        stats_path = Path(dataset.statistics_file) if dataset.statistics_file else None
        if stats_path and stats_path.exists():
            stats_data = json.loads(stats_path.read_text())
            stats = DatasetStatistics(**stats_data)
        else:
            stats = DatasetStatistics(dataset_id=dataset_id)

        validation = DatasetValidation(
            dataset_id=dataset_id,
            passed=dataset.validation_status == "passed",
            total_checks=12,
            passed_checks=12 if dataset.validation_status == "passed" else 0,
            failed_checks=0 if dataset.validation_status == "passed" else 12,
        )

        return self._quality.assess(stats, validation)

    # ------------------------------------------------------------------
    # Version management
    # ------------------------------------------------------------------

    def create_version(
        self,
        dataset_id: str,
        version: str | None = None,
        change_log: str = "",
    ) -> DatasetVersion:
        """Create a new version for a registered dataset."""
        dataset = self._registry.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")
        return self._versioning.create_version(dataset_id, version, change_log)

    def get_version(self, dataset_id: str, version: str) -> DatasetVersion | None:
        return self._versioning.get_version(dataset_id, version)

    def list_versions(self, dataset_id: str) -> list[DatasetVersion]:
        return self._versioning.list_versions(dataset_id)

    def get_latest_version(self, dataset_id: str) -> DatasetVersion | None:
        return self._versioning.get_latest_version(dataset_id)

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def split_dataset(
        self,
        dataset_id: str,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = False,
    ) -> DatasetSplit:
        """Split a registered dataset into train/validation/test."""
        dataset = self._registry.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")

        images = self._loader.load_images(dm_config.raw_dir)
        annotations = self._loader.load_annotations(dm_config.annotated_dir)

        if not images:
            raise SplitError(f"No images found for dataset: {dataset_id}")

        split = self._splitter.split(
            images=images,
            annotations=annotations,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratified=stratified,
        )

        split = DatasetSplit(
            dataset_id=dataset_id,
            train_images=split.train_images,
            validation_images=split.validation_images,
            test_images=split.test_images,
            train_annotations=split.train_annotations,
            validation_annotations=split.validation_annotations,
            test_annotations=split.test_annotations,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratified=stratified,
        )

        logger.info(
            "Split created: %s (train=%d, val=%d, test=%d)",
            dataset_id,
            len(split.train_images),
            len(split.validation_images),
            len(split.test_images),
        )
        return split

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_dataset(
        self,
        dataset_id: str,
        format_name: str = "yolo",
        output_dir: Path | None = None,
    ) -> DatasetExport:
        """Export a dataset to the specified format."""
        dataset = self._registry.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(f"Dataset not found: {dataset_id}")

        exporter = self._exporters.get(format_name)
        if exporter is None:
            raise ExportError(
                f"Unsupported export format: {format_name}. "
                f"Supported: {list(self._exporters.keys())}",
            )

        images = self._loader.load_images(dm_config.raw_dir)
        annotations = self._loader.load_annotations(dm_config.annotated_dir)
        output = output_dir or dm_config.exports_dir / format_name

        export = exporter.export(
            images=images,
            annotations=annotations,
            output_dir=output,
            class_mapping={cls: i for i, cls in enumerate(dataset.classes)},
        )

        export = DatasetExport(
            dataset_id=dataset_id,
            format_name=export.format_name,
            output_dir=export.output_dir,
            image_count=export.image_count,
            annotation_count=export.annotation_count,
            class_mapping=export.class_mapping,
        )

        logger.info(
            "Dataset exported: %s → %s (%s)",
            dataset_id, format_name, output,
        )
        return export

    # ------------------------------------------------------------------
    # Registry access
    # ------------------------------------------------------------------

    def register_dataset(self, info: DatasetInfo) -> DatasetInfo:
        return self._registry.register(info)

    def get_dataset(self, dataset_id: str) -> DatasetInfo | None:
        return self._registry.get(dataset_id)

    def list_datasets(self) -> list[DatasetInfo]:
        return self._registry.list_datasets()

    def update_dataset(self, info: DatasetInfo) -> DatasetInfo:
        return self._registry.update(info)

    def delete_dataset(self, dataset_id: str) -> bool:
        return self._registry.delete(dataset_id)
