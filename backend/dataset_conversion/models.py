from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class GeometryType(str, Enum):
    BBOX = "bbox"
    POLYGON = "polygon"
    OBB = "obb"
    SEGMENTATION = "segmentation"
    NORMALIZED = "normalized"


class CoordinateSystem(str, Enum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"


class DatasetFormat(str, Enum):
    COCO_JSON = "coco_json"
    YOLO_TXT = "yolo_txt"
    PASCAL_VOC = "pascal_voc"
    OPEN_IMAGES_CSV = "open_images_csv"
    GEOJSON = "geojson"
    CANONICAL = "canonical"
    CUSTOM = "custom"


class SplitType(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SplitStrategy(str, Enum):
    RANDOM = "random"
    STRATIFIED = "stratified"
    CLASS_BALANCED = "class_balanced"


class SourceCategory(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int | str = Field(..., description="Original category ID from the source dataset")
    name: str = Field(..., min_length=1, description="Original category name")
    supercategory: str | None = Field(None, description="Optional supercategory")


class SourceAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str | int = Field(..., description="Original annotation ID from the source dataset")
    image_id: str | int = Field(..., description="Reference to the source image ID")
    category_id: int | str = Field(..., description="Reference to the source category ID")
    category_name: str = Field(..., min_length=1, description="Original label string")
    geometry_type: GeometryType = Field(..., description="Type of geometry in the source")
    coordinates: tuple[float, ...] = Field(..., min_length=4, description="Raw coordinate values")
    coordinate_system: CoordinateSystem = Field(
        CoordinateSystem.PIXEL, description="Coordinate system of the geometry"
    )
    confidence: float | None = Field(None, ge=0.0, le=1.0, description="Optional confidence score")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional metadata preserved from source"
    )
    image_width: int | None = Field(None, ge=1, description="Width of the source image")
    image_height: int | None = Field(None, ge=1, description="Height of the source image")


class ImageInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1, description="Unique image identifier")
    file_path: str = Field(
        ..., min_length=1, description="Absolute or relative path to the image file"
    )
    width: int | None = Field(None, ge=1, description="Image width in pixels")
    height: int | None = Field(None, ge=1, description="Image height in pixels")
    format: str | None = Field(None, description="Image format (png, jpg, etc.)")
    color_space: str | None = Field(None, description="Color space (rgb, grayscale, etc.)")
    file_size_bytes: int | None = Field(None, ge=0, description="File size in bytes")
    metadata: dict[str, str] = Field(default_factory=dict, description="Additional image metadata")


class CanonicalAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique canonical annotation ID"
    )
    image_id: str = Field(..., min_length=1, description="Reference to the canonical image ID")
    canonical_label: str = Field(..., min_length=1, description="DSS ontology canonical value")
    canonical_name: str = Field(..., min_length=1, description="Human-readable canonical name")
    geometry_type: GeometryType = Field(..., description="Type of geometry")
    x: float = Field(..., description="X coordinate of the geometry")
    y: float = Field(..., description="Y coordinate of the geometry")
    width: float = Field(..., ge=0, description="Width of the geometry")
    height: float = Field(..., ge=0, description="Height of the geometry")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Confidence score (defaults to 1.0)")
    source_annotation_id: str | int | None = Field(
        None, description="Original annotation ID for provenance"
    )
    source_label: str | None = Field(None, description="Original source label for provenance")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional annotation metadata"
    )


class ConversionReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_images: int = Field(..., ge=0, description="Total images processed")
    total_annotations: int = Field(..., ge=0, description="Total annotations processed")
    mapped_annotations: int = Field(
        ..., ge=0, description="Annotations successfully mapped to ontology"
    )
    unmapped_annotations: int = Field(..., ge=0, description="Annotations that could not be mapped")
    skipped_images: int = Field(..., ge=0, description="Images skipped due to errors")
    errors: tuple[str, ...] = Field(
        default_factory=tuple, description="Error messages during conversion"
    )


class LoadResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., min_length=1, description="Name of the loaded dataset")
    source_path: str = Field(..., min_length=1, description="Path to the source dataset")
    dataset_format: str = Field(..., min_length=1, description="Format identifier")
    images: tuple[ImageInfo, ...] = Field(default_factory=tuple, description="Loaded images")
    annotations: tuple[SourceAnnotation, ...] = Field(
        default_factory=tuple, description="Loaded source annotations"
    )
    categories: tuple[SourceCategory, ...] = Field(
        default_factory=tuple, description="Loaded source categories"
    )
    image_count: int = Field(..., ge=0, description="Total number of images")
    annotation_count: int = Field(..., ge=0, description="Total number of annotations")
    category_count: int = Field(..., ge=0, description="Total number of categories")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional dataset metadata"
    )


class ConvertResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: CanonicalDataset = Field(..., description="The converted canonical dataset")
    report: ConversionReport = Field(..., description="Summary of the conversion process")


class MergeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: str = Field("sequential", description="Merge strategy: sequential, interleave")
    resolve_conflicts: bool = Field(
        True, description="Whether to resolve ontology conflicts during merge"
    )
    deduplicate_images: bool = Field(
        True, description="Whether to deduplicate images with the same content hash"
    )
    image_id_prefix: str = Field("merged", min_length=1, description="Prefix for merged image IDs")
    annotation_id_prefix: str = Field(
        "merged_ann", min_length=1, description="Prefix for merged annotation IDs"
    )


class MergeResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset: CanonicalDataset = Field(..., description="The merged canonical dataset")
    total_images: int = Field(..., ge=0, description="Total images after merge")
    total_annotations: int = Field(..., ge=0, description="Total annotations after merge")
    deduplicated_count: int = Field(0, ge=0, description="Number of deduplicated entries")
    source_datasets: tuple[str, ...] = Field(
        default_factory=tuple, description="Names of source datasets"
    )


class SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy: SplitStrategy = Field(SplitStrategy.RANDOM, description="Split strategy")
    train_ratio: float = Field(0.7, gt=0.0, lt=1.0, description="Ratio for training set")
    val_ratio: float = Field(0.15, ge=0.0, lt=0.5, description="Ratio for validation set")
    test_ratio: float = Field(0.15, ge=0.0, lt=0.5, description="Ratio for test set")
    seed: int = Field(42, ge=0, description="Random seed for reproducibility")
    shuffle: bool = Field(True, description="Whether to shuffle before splitting")
    stratify_by: str | None = Field(None, description="Field to stratify by (class, etc.)")


class SplitResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    train: CanonicalDataset = Field(..., description="Training split")
    val: CanonicalDataset = Field(..., description="Validation split")
    test: CanonicalDataset = Field(..., description="Test split")
    train_ratio: float = Field(..., ge=0.0, le=1.0, description="Actual training ratio")
    val_ratio: float = Field(..., ge=0.0, le=1.0, description="Actual validation ratio")
    test_ratio: float = Field(..., ge=0.0, le=1.0, description="Actual test ratio")


class ValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    valid: bool = Field(..., description="Whether the dataset passed validation")
    total_checks: int = Field(..., ge=0, description="Total number of validation checks performed")
    passed_checks: int = Field(..., ge=0, description="Number of checks that passed")
    failed_checks: int = Field(..., ge=0, description="Number of checks that failed")
    errors: tuple[str, ...] = Field(default_factory=tuple, description="Validation error messages")
    warnings: tuple[str, ...] = Field(default_factory=tuple, description="Validation warnings")


class DatasetStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_images: int = Field(..., ge=0, description="Total number of images")
    total_annotations: int = Field(..., ge=0, description="Total number of annotations")
    total_classes: int = Field(..., ge=0, description="Total number of unique classes")
    classes: tuple[tuple[str, int], ...] = Field(
        default_factory=tuple, description="(class_name, count) for each class"
    )
    images_with_annotations: int = Field(
        ..., ge=0, description="Images that have at least one annotation"
    )
    images_without_annotations: int = Field(..., ge=0, description="Images with zero annotations")
    avg_annotations_per_image: float = Field(
        ..., ge=0.0, description="Average annotations per image"
    )
    min_annotations_per_image: int = Field(
        ..., ge=0, description="Minimum annotations on any image"
    )
    max_annotations_per_image: int = Field(
        ..., ge=0, description="Maximum annotations on any image"
    )
    avg_image_width: float = Field(..., ge=0.0, description="Average image width")
    avg_image_height: float = Field(..., ge=0.0, description="Average image height")
    avg_bbox_width: float = Field(..., ge=0.0, description="Average bounding box width")
    avg_bbox_height: float = Field(..., ge=0.0, description="Average bounding box height")
    class_balance: dict[str, float] = Field(
        default_factory=dict, description="Class name to proportion mapping"
    )
    coverage_report: dict[str, object] = Field(
        default_factory=dict, description="Coverage statistics"
    )


class DatasetManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    manifest_id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique manifest identifier"
    )
    dataset_version: str = Field(..., min_length=1, description="Version of the dataset")
    ontology_version: str = Field(..., min_length=1, description="Version of the ontology used")
    source_datasets: tuple[str, ...] = Field(
        default_factory=tuple, description="Source dataset names"
    )
    conversion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When conversion occurred"
    )
    statistics: DatasetStatistics | None = Field(None, description="Computed dataset statistics")
    checksums: dict[str, str] = Field(
        default_factory=dict, description="File path to checksum mapping"
    )
    pipeline_version: str = Field(
        ..., min_length=1, description="Version of the conversion pipeline"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional manifest metadata"
    )


class ExportResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    export_format: str = Field(..., description="Format exported to")
    output_path: str = Field(..., min_length=1, description="Path where the export was written")
    images_exported: int = Field(..., ge=0, description="Number of images exported")
    annotations_exported: int = Field(..., ge=0, description="Number of annotations exported")
    file_count: int = Field(..., ge=0, description="Number of files written")
    file_size_bytes: int = Field(..., ge=0, description="Total size of exported files in bytes")


class CanonicalDataset(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique dataset identifier")
    name: str = Field(..., min_length=1, description="Dataset name")
    images: tuple[ImageInfo, ...] = Field(
        default_factory=tuple, description="Images in the dataset"
    )
    annotations: tuple[CanonicalAnnotation, ...] = Field(
        default_factory=tuple, description="Annotations in the dataset"
    )
    image_count: int = Field(..., ge=0, description="Total number of images")
    annotation_count: int = Field(..., ge=0, description="Total number of annotations")
    class_count: int = Field(..., ge=0, description="Total number of unique classes")
    ontology_version: str = Field(
        "1.0.0", min_length=1, description="Ontology version used for mapping"
    )
    pipeline_version: str = Field(
        "1.0.0", min_length=1, description="Pipeline version used for conversion"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Dataset creation timestamp"
    )
    source_datasets: tuple[str, ...] = Field(
        default_factory=tuple, description="Source dataset names"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional dataset metadata"
    )
