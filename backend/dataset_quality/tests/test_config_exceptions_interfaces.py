from __future__ import annotations

import pytest

from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.exceptions import (
    AnnotationValidationError,
    ClassValidationError,
    ConsistencyCheckError,
    CoverageAnalysisError,
    DuplicateDetectionError,
    GeometryValidationError,
    ImageValidationError,
    ImbalanceAnalysisError,
    IntegrityCheckError,
    OutlierDetectionError,
    PipelineError,
    QualityError,
    ReportGenerationError,
    ScoringError,
)


class TestConfig:
    def test_defaults(self) -> None:
        cfg = DatasetQualityConfig()
        assert cfg.min_image_width == 32
        assert cfg.min_image_height == 32
        assert cfg.max_image_width == 10000
        assert cfg.max_image_height == 10000
        assert cfg.min_object_area == 4.0
        assert cfg.max_aspect_ratio == 20.0
        assert cfg.rare_class_threshold == 0.01
        assert cfg.duplicate_iou_threshold == 0.95
        assert cfg.near_duplicate_iou_threshold == 0.85
        assert cfg.outlier_std_dev_threshold == 3.0
        assert cfg.min_class_samples == 5
        assert cfg.output_dir == "quality_reports"
        assert cfg.strict_mode is False
        assert cfg.pipeline_version == "1.0.0"

    def test_env_prefix(self) -> None:
        cfg = DatasetQualityConfig(_env_prefix="DATASET_QUALITY_")
        assert cfg.model_config.get("env_prefix") == "DATASET_QUALITY_"

    def test_singleton(self) -> None:
        assert dataset_quality_config is not None
        assert isinstance(dataset_quality_config, DatasetQualityConfig)

    def test_custom_values(self) -> None:
        cfg = DatasetQualityConfig(
            min_image_width=64,
            max_image_width=8000,
            rare_class_threshold=0.05,
            strict_mode=True,
        )
        assert cfg.min_image_width == 64
        assert cfg.max_image_width == 8000
        assert cfg.rare_class_threshold == 0.05
        assert cfg.strict_mode is True


class TestExceptions:
    def test_base_exception(self) -> None:
        assert issubclass(QualityError, Exception)
        with pytest.raises(QualityError):
            raise QualityError("base error")

    def test_image_validation_error(self) -> None:
        assert issubclass(ImageValidationError, QualityError)
        with pytest.raises(ImageValidationError):
            raise ImageValidationError("bad image")

    def test_annotation_validation_error(self) -> None:
        assert issubclass(AnnotationValidationError, QualityError)
        with pytest.raises(AnnotationValidationError):
            raise AnnotationValidationError("bad annotation")

    def test_class_validation_error(self) -> None:
        assert issubclass(ClassValidationError, QualityError)
        with pytest.raises(ClassValidationError):
            raise ClassValidationError("bad class")

    def test_geometry_validation_error(self) -> None:
        assert issubclass(GeometryValidationError, QualityError)
        with pytest.raises(GeometryValidationError):
            raise GeometryValidationError("bad geometry")

    def test_duplicate_detection_error(self) -> None:
        assert issubclass(DuplicateDetectionError, QualityError)
        with pytest.raises(DuplicateDetectionError):
            raise DuplicateDetectionError("dup error")

    def test_outlier_detection_error(self) -> None:
        assert issubclass(OutlierDetectionError, QualityError)
        with pytest.raises(OutlierDetectionError):
            raise OutlierDetectionError("outlier error")

    def test_imbalance_analysis_error(self) -> None:
        assert issubclass(ImbalanceAnalysisError, QualityError)

    def test_coverage_analysis_error(self) -> None:
        assert issubclass(CoverageAnalysisError, QualityError)

    def test_consistency_check_error(self) -> None:
        assert issubclass(ConsistencyCheckError, QualityError)

    def test_integrity_check_error(self) -> None:
        assert issubclass(IntegrityCheckError, QualityError)

    def test_scoring_error(self) -> None:
        assert issubclass(ScoringError, QualityError)

    def test_report_generation_error(self) -> None:
        assert issubclass(ReportGenerationError, QualityError)

    def test_pipeline_error(self) -> None:
        assert issubclass(PipelineError, QualityError)

    def test_image_validation_error_message(self) -> None:
        with pytest.raises(ImageValidationError, match="bad"):
            raise ImageValidationError("bad image")

    def test_pipeline_error_message(self) -> None:
        with pytest.raises(PipelineError, match="fail"):
            raise PipelineError("pipeline fail")

    def test_exception_hierarchy_all_quality_error(self) -> None:
        for exc in [
            ImageValidationError,
            AnnotationValidationError,
            ClassValidationError,
            GeometryValidationError,
            DuplicateDetectionError,
            OutlierDetectionError,
            ImbalanceAnalysisError,
            CoverageAnalysisError,
            ConsistencyCheckError,
            IntegrityCheckError,
            ScoringError,
            ReportGenerationError,
            PipelineError,
        ]:
            assert issubclass(exc, QualityError)
