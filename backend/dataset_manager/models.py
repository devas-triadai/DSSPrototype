"""Strongly typed Pydantic models for the Dataset Management Platform.

Every model uses frozen=True for immutability, following the DSS contract pattern.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class DatasetLicense(BaseModel):
    """License information for a dataset."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="Proprietary — demonstration use only")
    url: str | None = Field(default=None)
    attribution: str | None = Field(default=None)


class DatasetChecksum(BaseModel):
    """Checksum data for integrity verification."""

    model_config = ConfigDict(frozen=True)

    algorithm: str = Field(default="sha256")
    value: str = Field(default="")
    file_path: str = Field(default="")
    verified_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DatasetInfo(BaseModel):
    """Core registry entry for a dataset.

    Every dataset in the system is tracked via this record.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., description="Unique registry identifier")
    dataset_name: str = Field(..., description="Human-readable name")
    dataset_version: str = Field(default="1.0.0")
    dataset_type: str = Field(default="raw", description="raw | annotated | split")
    description: str = Field(default="")
    created_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = Field(default="")
    license: DatasetLicense = Field(default_factory=DatasetLicense)
    image_count: int = Field(default=0)
    annotation_count: int = Field(default=0)
    class_count: int = Field(default=0)
    classes: list[str] = Field(default_factory=list)
    supported_formats: list[str] = Field(default_factory=lambda: ["yolo", "coco", "voc"])
    checksum: DatasetChecksum | None = Field(default=None)
    validation_status: str = Field(default="pending", description="pending | passed | failed")
    quality_score: float = Field(default=0.0)
    statistics_file: str = Field(default="")
    metadata_file: str = Field(default="")


class DatasetVersion(BaseModel):
    """A semantic version record for a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    version: str = Field(default="1.0.0")
    parent_version: str | None = Field(default=None)
    change_log: str = Field(default="")
    created_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checksum: DatasetChecksum | None = Field(default=None)
    validation_report: str = Field(default="")
    statistics_file: str = Field(default="")
    metadata_file: str = Field(default="")


class DatasetMetadata(BaseModel):
    """Complete metadata record automatically generated per dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    dataset_name: str = Field(...)
    dataset_version: str = Field(default="1.0.0")
    created_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = Field(default="")
    license: DatasetLicense = Field(default_factory=DatasetLicense)
    description: str = Field(default="")
    image_count: int = Field(default=0)
    annotation_count: int = Field(default=0)
    class_count: int = Field(default=0)
    classes: list[str] = Field(default_factory=list)
    supported_formats: list[str] = Field(default_factory=lambda: ["yolo", "coco", "voc"])
    statistics: "DatasetStatistics | None" = Field(default=None)
    quality: "DatasetQuality | None" = Field(default=None)
    validation: "DatasetValidation | None" = Field(default=None)
    checksum: DatasetChecksum | None = Field(default=None)
    splits: "DatasetSplit | None" = Field(default=None)


class DatasetStatistics(BaseModel):
    """Comprehensive statistics for a single dataset."""

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
    dataset_completeness: float = Field(default=0.0)
    coverage_score: float = Field(default=0.0)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DatasetQuality(BaseModel):
    """Quality assessment report for a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    annotation_completeness: float = Field(default=0.0)
    dataset_balance: float = Field(default=0.0)
    image_quality_metadata: float = Field(default=0.0)
    duplicate_percentage: float = Field(default=0.0)
    missing_labels: int = Field(default=0)
    validation_score: float = Field(default=0.0)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DatasetValidation(BaseModel):
    """Validation report for a dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    passed: bool = Field(default=False)
    missing_images: list[str] = Field(default_factory=list)
    missing_labels: list[str] = Field(default_factory=list)
    empty_annotations: list[str] = Field(default_factory=list)
    duplicate_images: list[str] = Field(default_factory=list)
    duplicate_annotations: list[str] = Field(default_factory=list)
    corrupted_files: list[str] = Field(default_factory=list)
    unsupported_extensions: list[str] = Field(default_factory=list)
    invalid_bounding_boxes: list[str] = Field(default_factory=list)
    negative_coordinates: list[str] = Field(default_factory=list)
    class_mismatches: list[str] = Field(default_factory=list)
    orphan_labels: list[str] = Field(default_factory=list)
    invalid_metadata: list[str] = Field(default_factory=list)
    total_checks: int = Field(default=0)
    passed_checks: int = Field(default=0)
    failed_checks: int = Field(default=0)
    validated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DatasetSplit(BaseModel):
    """Train/Validation/Test split definition."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    train_images: list[str] = Field(default_factory=list)
    validation_images: list[str] = Field(default_factory=list)
    test_images: list[str] = Field(default_factory=list)
    train_annotations: list[str] = Field(default_factory=list)
    validation_annotations: list[str] = Field(default_factory=list)
    test_annotations: list[str] = Field(default_factory=list)
    train_ratio: float = Field(default=0.7)
    validation_ratio: float = Field(default=0.15)
    test_ratio: float = Field(default=0.15)
    seed: int = Field(default=42)
    stratified: bool = Field(default=False)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DatasetExport(BaseModel):
    """Metadata for a completed dataset export."""

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(...)
    format_name: str = Field(...)
    output_dir: str = Field(...)
    image_count: int = Field(default=0)
    annotation_count: int = Field(default=0)
    class_mapping: dict[str, int] = Field(default_factory=dict)
    checksum: DatasetChecksum | None = Field(default=None)
    exported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
