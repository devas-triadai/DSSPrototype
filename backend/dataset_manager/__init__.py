"""Dataset Management Platform — single source of truth for all CV training data.

Manages raw datasets, annotated datasets, versions, metadata, validation,
statistics, quality assessment, exports, and train/validation/test splits.
"""

from backend.dataset_manager.checksum import ChecksumGenerator
from backend.dataset_manager.config import DatasetManagerConfig, dm_config
from backend.dataset_manager.exporter import (
    CocoExporter,
    DatasetExporter,
    PascalVocExporter,
    YoloExporter,
)
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
    DatasetChecksum,
    DatasetExport,
    DatasetInfo,
    DatasetLicense,
    DatasetMetadata,
    DatasetQuality,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidation,
    DatasetVersion,
)
from backend.dataset_manager.quality import QualityEngine
from backend.dataset_manager.registry import DatasetRegistry
from backend.dataset_manager.service import DatasetManagementService
from backend.dataset_manager.splitter import DatasetSplitter
from backend.dataset_manager.statistics import StatisticsEngine
from backend.dataset_manager.validator import DatasetValidator
from backend.dataset_manager.versioning import DatasetVersioning

__all__ = [
    "DatasetManagerConfig",
    "dm_config",
    "DatasetInfo",
    "DatasetVersion",
    "DatasetMetadata",
    "DatasetStatistics",
    "DatasetQuality",
    "DatasetExport",
    "DatasetSplit",
    "DatasetValidation",
    "DatasetLicense",
    "DatasetChecksum",
    "RegistryInterface",
    "DatasetLoaderInterface",
    "ValidationEngineInterface",
    "StatisticsEngineInterface",
    "QualityEngineInterface",
    "VersioningInterface",
    "SplitterInterface",
    "DatasetExporterInterface",
    "MetadataGeneratorInterface",
    "ChecksumInterface",
    "DatasetRegistry",
    "DatasetValidator",
    "StatisticsEngine",
    "DatasetSplitter",
    "DatasetVersioning",
    "DatasetExporter",
    "YoloExporter",
    "CocoExporter",
    "PascalVocExporter",
    "QualityEngine",
    "MetadataGenerator",
    "DatasetLoader",
    "ChecksumGenerator",
    "DatasetManagementService",
]
