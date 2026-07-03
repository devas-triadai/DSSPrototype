from __future__ import annotations

from backend.dataset_conversion.annotation_converter import AnnotationConverter
from backend.dataset_conversion.annotation_loader import AnnotationLoader

# Config
from backend.dataset_conversion.config import (
    DatasetConversionConfig,
    dataset_conversion_config,
)
from backend.dataset_conversion.conversion_pipeline import ConversionPipeline
from backend.dataset_conversion.dataset_exporter import DatasetExporter

# Core components
from backend.dataset_conversion.dataset_loader import DatasetLoader
from backend.dataset_conversion.dataset_merger import DatasetMerger
from backend.dataset_conversion.dataset_splitter import DatasetSplitter
from backend.dataset_conversion.dataset_statistics import DatasetStatisticsGenerator
from backend.dataset_conversion.dataset_validator import DatasetValidator

# Exceptions
from backend.dataset_conversion.exceptions import (
    AnnotationError,
    ConversionError,
    ExportError,
    GeometryError,
    ImageError,
    LoadError,
    ManifestError,
    MergeError,
    OntologyAdapterError,
    PipelineError,
    SplitError,
    StatisticsError,
    ValidationError,
)
from backend.dataset_conversion.geometry_converter import GeometryConverter
from backend.dataset_conversion.image_converter import ImageConverter

# Interfaces
from backend.dataset_conversion.interfaces import (
    AnnotationConverterInterface,
    AnnotationLoaderInterface,
    ConversionPipelineInterface,
    DatasetConversionServiceInterface,
    DatasetExporterInterface,
    DatasetLoaderInterface,
    DatasetMergerInterface,
    DatasetSplitterInterface,
    DatasetStatisticsInterface,
    DatasetValidatorInterface,
    GeometryConverterInterface,
    ImageConverterInterface,
    ManifestBuilderInterface,
    OntologyAdapterInterface,
)
from backend.dataset_conversion.manifest_builder import ManifestBuilder

# Enums
# Models
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    ConversionReport,
    ConvertResult,
    CoordinateSystem,
    DatasetFormat,
    DatasetManifest,
    DatasetStatistics,
    ExportResult,
    GeometryType,
    ImageInfo,
    LoadResult,
    MergeConfig,
    MergeResult,
    SourceAnnotation,
    SourceCategory,
    SplitConfig,
    SplitResult,
    SplitStrategy,
    SplitType,
    ValidationReport,
)
from backend.dataset_conversion.ontology_adapter import OntologyAdapter

# Service
from backend.dataset_conversion.service import DatasetConversionService

__all__ = [
    # Config
    "DatasetConversionConfig",
    "dataset_conversion_config",
    # Exceptions
    "ConversionError",
    "LoadError",
    "AnnotationError",
    "OntologyAdapterError",
    "GeometryError",
    "ImageError",
    "MergeError",
    "SplitError",
    "ValidationError",
    "StatisticsError",
    "ManifestError",
    "ExportError",
    "PipelineError",
    # Enums
    "GeometryType",
    "CoordinateSystem",
    "DatasetFormat",
    "SplitType",
    "SplitStrategy",
    # Models
    "ImageInfo",
    "SourceAnnotation",
    "CanonicalAnnotation",
    "SourceCategory",
    "CanonicalDataset",
    "LoadResult",
    "ConvertResult",
    "ConversionReport",
    "MergeConfig",
    "MergeResult",
    "SplitConfig",
    "SplitResult",
    "ValidationReport",
    "DatasetStatistics",
    "DatasetManifest",
    "ExportResult",
    # Interfaces
    "DatasetLoaderInterface",
    "AnnotationLoaderInterface",
    "OntologyAdapterInterface",
    "GeometryConverterInterface",
    "AnnotationConverterInterface",
    "ImageConverterInterface",
    "DatasetMergerInterface",
    "DatasetSplitterInterface",
    "DatasetValidatorInterface",
    "DatasetStatisticsInterface",
    "ManifestBuilderInterface",
    "DatasetExporterInterface",
    "ConversionPipelineInterface",
    "DatasetConversionServiceInterface",
    # Core components
    "DatasetLoader",
    "AnnotationLoader",
    "OntologyAdapter",
    "GeometryConverter",
    "AnnotationConverter",
    "ImageConverter",
    "DatasetMerger",
    "DatasetSplitter",
    "DatasetValidator",
    "DatasetStatisticsGenerator",
    "ManifestBuilder",
    "DatasetExporter",
    "ConversionPipeline",
    # Service
    "DatasetConversionService",
]
