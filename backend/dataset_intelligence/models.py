"""Strongly typed Pydantic models for the Dataset Intelligence Pipeline.

Every model uses frozen=True for immutability, following the DSS contract pattern.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class Annotation(BaseModel):
    """A single object detection annotation."""

    model_config = ConfigDict(frozen=True)

    class_name: str = Field(..., description="Original class label")
    normalized_class: str = Field(default="", description="After normalization")
    ontology_class: str = Field(default="", description="After ontology mapping")
    bbox: tuple[float, float, float, float] = Field(
        ...,
        description="x_min, y_min, x_max, y_max in normalized coordinates",
    )
    bbox_format: str = Field(default="xyxy_normalized")
    confidence: float | None = Field(default=None)
    segmentation: list[tuple[float, float]] | None = Field(default=None)
    attributes: dict[str, str] = Field(default_factory=dict)


class ImageRecord(BaseModel):
    """A single image with its annotations and metadata."""

    model_config = ConfigDict(frozen=True)

    image_id: str = Field(...)
    image_path: str = Field(...)
    image_name: str = Field(...)
    width: int = Field(default=0)
    height: int = Field(default=0)
    channels: int = Field(default=3)
    format: str = Field(default="")
    annotations: list[Annotation] = Field(default_factory=list)
    checksum: str = Field(default="")
    metadata: dict[str, object] = Field(default_factory=dict)
    provenance: "ProvenanceRecord | None" = Field(default=None)


class ProvenanceRecord(BaseModel):
    """Provenance tracking for every image."""

    model_config = ConfigDict(frozen=True)

    source_dataset: str = Field(...)
    original_class: str = Field(default="")
    normalized_class: str = Field(default="")
    ontology_class: str = Field(default="")
    import_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    license: str = Field(default="")
    dataset_version: str = Field(default="")
    checksum: str = Field(default="")
    original_path: str = Field(default="")
    import_format: str = Field(default="")


class RawDataset(BaseModel):
    """Dataset as parsed from an import format."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    import_format: str = Field(...)
    source_path: str = Field(...)
    images: list[ImageRecord] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ValidationReport(BaseModel):
    """Validation report for a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    passed: bool = Field(default=False)
    missing_images: list[str] = Field(default_factory=list)
    missing_annotations: list[str] = Field(default_factory=list)
    empty_annotations: list[str] = Field(default_factory=list)
    corrupted_files: list[str] = Field(default_factory=list)
    unsupported_extensions: list[str] = Field(default_factory=list)
    invalid_bounding_boxes: list[str] = Field(default_factory=list)
    negative_coordinates: list[str] = Field(default_factory=list)
    class_mismatches: list[str] = Field(default_factory=list)
    orphan_labels: list[str] = Field(default_factory=list)
    total_checks: int = Field(default=0)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    validated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class NormalizedDataset(BaseModel):
    """Dataset after normalization."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    images: list[ImageRecord] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    class_mapping: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)
    normalization_log: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OntologyMappingReport(BaseModel):
    """Report of ontology mapping applied to a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    mappings: dict[str, str] = Field(default_factory=dict)
    unmapped_classes: list[str] = Field(default_factory=list)
    ontology_coverage: float = Field(default=0.0)
    ontology_version: str = Field(default="")
    mapped_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DuplicateEntry(BaseModel):
    """A single duplicate detection entry."""

    model_config = ConfigDict(frozen=True)

    duplicate_type: str = Field(..., description="image | annotation | hash | near_duplicate")
    original_id: str = Field(...)
    duplicate_id: str = Field(...)
    similarity_score: float = Field(default=1.0)
    reason: str = Field(default="")


class DuplicateReport(BaseModel):
    """Duplicate detection report."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    duplicates: list[DuplicateEntry] = Field(default_factory=list)
    duplicate_image_count: int = Field(default=0)
    duplicate_annotation_count: int = Field(default=0)
    duplicate_ratio: float = Field(default=0.0)
    detected_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HarmonizedDataset(BaseModel):
    """Dataset after class harmonization."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    images: list[ImageRecord] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    harmonization_mapping: dict[str, str] = Field(default_factory=dict)
    ontology_version: str = Field(default="")
    metadata: dict[str, object] = Field(default_factory=dict)
    harmonized_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MergedDataset(BaseModel):
    """Dataset after merging multiple datasets."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    source_datasets: list[str] = Field(default_factory=list)
    images: list[ImageRecord] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    provenance: list[ProvenanceRecord] = Field(default_factory=list)
    merged_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class QualityReport(BaseModel):
    """Quality assessment report."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_ratio: float = Field(default=0.0)
    class_imbalance_ratio: float = Field(default=0.0)
    missing_annotations: int = Field(default=0)
    missing_images: int = Field(default=0)
    annotation_consistency: float = Field(default=0.0)
    ontology_coverage: float = Field(default=0.0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    passed: bool = Field(default=False)
    assessed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class StatisticsReport(BaseModel):
    """Comprehensive statistics for a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    total_images: int = Field(default=0)
    total_annotations: int = Field(default=0)
    classes: list[str] = Field(default_factory=list)
    class_count: int = Field(default=0)
    objects_per_class: dict[str, int] = Field(default_factory=dict)
    class_distribution: dict[str, float] = Field(default_factory=dict)
    class_imbalance_ratio: float = Field(default=0.0)
    average_objects_per_image: float = Field(default=0.0)
    average_image_width: float = Field(default=0.0)
    average_image_height: float = Field(default=0.0)
    resolution_distribution: dict[str, int] = Field(default_factory=dict)
    bbox_width_distribution: dict[str, int] = Field(default_factory=dict)
    bbox_height_distribution: dict[str, int] = Field(default_factory=dict)
    aspect_ratio_distribution: dict[str, int] = Field(default_factory=dict)
    ontology_coverage: float = Field(default=0.0)
    duplicate_ratio: float = Field(default=0.0)
    dataset_diversity: float = Field(default=0.0)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExportResult(BaseModel):
    """Result of a dataset export."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    format_name: str = Field(...)
    output_dir: str = Field(...)
    image_count: int = Field(default=0)
    annotation_count: int = Field(default=0)
    class_mapping: dict[str, int] = Field(default_factory=dict)
    exported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ImportResult(BaseModel):
    """Result of a dataset import operation."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    import_format: str = Field(...)
    source_path: str = Field(...)
    raw_dataset: RawDataset = Field(...)
    validation_report: ValidationReport | None = Field(default=None)
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = Field(default="pending", description="pending | validated | rejected | processed")


class ProcessedDataset(BaseModel):
    """Final processed dataset ready for training or registration."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    version: str = Field(default="1.0.0")
    images: list[ImageRecord] = Field(default_factory=list)
    classes: list[str] = Field(default_factory=list)
    class_mapping: dict[str, int] = Field(default_factory=dict)
    splits: dict[str, list[str]] = Field(default_factory=dict)
    statistics: StatisticsReport | None = Field(default=None)
    quality: QualityReport | None = Field(default=None)
    validation: ValidationReport | None = Field(default=None)
    duplicates: DuplicateReport | None = Field(default=None)
    ontology_mapping: OntologyMappingReport | None = Field(default=None)
    export_result: ExportResult | None = Field(default=None)
    metadata: dict[str, object] = Field(default_factory=dict)
    processed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = Field(default="ready", description="ready | rejected | failed")


class DatasetIntelligenceRegistryEntry(BaseModel):
    """Entry in the dataset intelligence registry."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    version: str = Field(default="1.0.0")
    import_format: str = Field(default="")
    status: str = Field(default="pending")
    quality_score: float = Field(default=0.0)
    processed_path: str = Field(default="")
    statistics_file: str = Field(default="")
    quality_file: str = Field(default="")
    validation_file: str = Field(default="")
    report_file: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
