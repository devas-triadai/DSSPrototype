from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class QualityCategory(str, Enum):
    IMAGE = "image"
    ANNOTATION = "annotation"
    CLASS = "class"
    GEOMETRY = "geometry"
    DUPLICATE = "duplicate"
    OUTLIER = "outlier"
    IMBALANCE = "imbalance"
    COVERAGE = "coverage"
    CONSISTENCY = "consistency"
    INTEGRITY = "integrity"


class LetterGrade(str, Enum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class QualityIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    severity: Severity = Field(..., description="Severity level")
    category: QualityCategory = Field(..., description="Quality category")
    message: str = Field(..., min_length=1, description="Human-readable description")
    location: str = Field("", description="Specific location (image ID, annotation ID, etc.)")
    suggestion: str | None = Field(None, description="Recommended fix")
    details: dict[str, object] = Field(
        default_factory=dict, description="Additional structured details"
    )


class ImageValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_images: int = Field(0, ge=0, description="Total images inspected")
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    corrupt_count: int = Field(0, ge=0)
    unreadable_count: int = Field(0, ge=0)
    wrong_format_count: int = Field(0, ge=0)
    wrong_color_space_count: int = Field(0, ge=0)
    tiny_image_count: int = Field(0, ge=0)
    oversized_image_count: int = Field(0, ge=0)
    blank_image_count: int = Field(0, ge=0)
    passed: bool = Field(True)


class AnnotationValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_annotations: int = Field(0, ge=0)
    total_images: int = Field(0, ge=0)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    missing_annotation_count: int = Field(0, ge=0)
    negative_coordinate_count: int = Field(0, ge=0)
    zero_area_count: int = Field(0, ge=0)
    out_of_bounds_count: int = Field(0, ge=0)
    invalid_polygon_count: int = Field(0, ge=0)
    broken_segmentation_count: int = Field(0, ge=0)
    broken_obb_count: int = Field(0, ge=0)
    images_without_annotations: tuple[str, ...] = Field(default_factory=tuple)
    passed: bool = Field(True)


class ClassValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_classes: int = Field(0, ge=0)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    unknown_class_count: int = Field(0, ge=0)
    ontology_mismatch_count: int = Field(0, ge=0)
    unused_class_count: int = Field(0, ge=0)
    rare_class_count: int = Field(0, ge=0)
    missing_required_class_count: int = Field(0, ge=0)
    imbalance_ratio: float = Field(1.0, ge=0.0)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    passed: bool = Field(True)


class GeometryValidationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_geometries: int = Field(0, ge=0)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    invalid_bbox_count: int = Field(0, ge=0)
    invalid_polygon_count: int = Field(0, ge=0)
    invalid_segmentation_count: int = Field(0, ge=0)
    invalid_rotated_box_count: int = Field(0, ge=0)
    invalid_normalized_coord_count: int = Field(0, ge=0)
    invalid_pixel_coord_count: int = Field(0, ge=0)
    passed: bool = Field(True)


class DuplicateDetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    duplicate_image_pairs: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    duplicate_annotation_pairs: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    near_duplicate_image_pairs: tuple[tuple[str, str], ...] = Field(default_factory=tuple)
    repeated_ids: tuple[str, ...] = Field(default_factory=tuple)
    total_duplicate_images: int = Field(0, ge=0)
    total_duplicate_annotations: int = Field(0, ge=0)
    passed: bool = Field(True)


class OutlierDetectionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    extreme_aspect_ratios: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    tiny_objects: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    huge_objects: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    suspicious_annotations: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    abnormal_resolutions: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    total_outliers: int = Field(0, ge=0)
    passed: bool = Field(True)


class ImbalanceAnalysisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    class_distribution: dict[str, int] = Field(default_factory=dict)
    sorted_distribution: tuple[dict[str, object], ...] = Field(default_factory=tuple)
    total_samples: int = Field(0, ge=0)
    num_classes: int = Field(0, ge=0)
    minority_classes: tuple[str, ...] = Field(default_factory=tuple)
    majority_classes: tuple[str, ...] = Field(default_factory=tuple)
    long_tail_ratio: float = Field(0.0, ge=0.0)
    imbalance_ratio: float = Field(1.0, ge=1.0)
    recommended_augmentations: dict[str, object] = Field(default_factory=dict)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    passed: bool = Field(True)


class CoverageAnalysisResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    ontology_coverage_pct: float = Field(0.0, ge=0.0, le=100.0)
    image_coverage_pct: float = Field(0.0, ge=0.0, le=100.0)
    scene_diversity: float = Field(0.0, ge=0.0)
    object_diversity: float = Field(0.0, ge=0.0)
    completeness_pct: float = Field(0.0, ge=0.0, le=100.0)
    covered_classes: int = Field(0, ge=0)
    total_ontology_classes: int = Field(0, ge=0)
    missing_classes: tuple[str, ...] = Field(default_factory=tuple)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    passed: bool = Field(True)


class ConsistencyCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata_consistent: bool = Field(True)
    split_consistent: bool = Field(True)
    ontology_consistent: bool = Field(True)
    annotation_consistent: bool = Field(True)
    version_consistent: bool = Field(True)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    passed: bool = Field(True)


class IntegrityCheckResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    checksums_valid: bool = Field(True)
    manifest_valid: bool = Field(True)
    all_files_present: bool = Field(True)
    no_broken_references: bool = Field(True)
    version_valid: bool = Field(True)
    missing_files: tuple[str, ...] = Field(default_factory=tuple)
    broken_references: tuple[str, ...] = Field(default_factory=tuple)
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    passed: bool = Field(True)


class ScoreBreakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_quality: float = Field(0.0, ge=0.0, le=100.0)
    annotation_quality: float = Field(0.0, ge=0.0, le=100.0)
    geometry_quality: float = Field(0.0, ge=0.0, le=100.0)
    ontology_coverage: float = Field(0.0, ge=0.0, le=100.0)
    balance: float = Field(0.0, ge=0.0, le=100.0)
    integrity: float = Field(0.0, ge=0.0, le=100.0)
    consistency: float = Field(0.0, ge=0.0, le=100.0)


class DatasetScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    overall: float = Field(0.0, ge=0.0, le=100.0)
    letter_grade: LetterGrade = Field(LetterGrade.F)
    production_ready: bool = Field(False)
    breakdown: ScoreBreakdown = Field(default_factory=lambda: ScoreBreakdown())
    issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)


class QualityReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., min_length=1)
    dataset_version: str = Field("1.0.0")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    pipeline_version: str = Field("1.0.0")
    overall_score: DatasetScore = Field(default_factory=lambda: DatasetScore())
    image_validation: ImageValidationResult | None = Field(None)
    annotation_validation: AnnotationValidationResult | None = Field(None)
    class_validation: ClassValidationResult | None = Field(None)
    geometry_validation: GeometryValidationResult | None = Field(None)
    duplicate_detection: DuplicateDetectionResult | None = Field(None)
    outlier_detection: OutlierDetectionResult | None = Field(None)
    imbalance_analysis: ImbalanceAnalysisResult | None = Field(None)
    coverage_analysis: CoverageAnalysisResult | None = Field(None)
    consistency_check: ConsistencyCheckResult | None = Field(None)
    integrity_check: IntegrityCheckResult | None = Field(None)
    all_issues: tuple[QualityIssue, ...] = Field(default_factory=tuple)
    error_count: int = Field(0, ge=0)
    warning_count: int = Field(0, ge=0)
    info_count: int = Field(0, ge=0)
    summary: str = Field("")
