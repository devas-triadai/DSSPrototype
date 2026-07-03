from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.dataset_quality.models import (
    AnnotationValidationResult,
    ClassValidationResult,
    ConsistencyCheckResult,
    CoverageAnalysisResult,
    DatasetScore,
    DuplicateDetectionResult,
    GeometryValidationResult,
    ImageValidationResult,
    ImbalanceAnalysisResult,
    IntegrityCheckResult,
    LetterGrade,
    OutlierDetectionResult,
    QualityCategory,
    QualityIssue,
    QualityReport,
    ScoreBreakdown,
    Severity,
)


class TestEnums:
    def test_severity_values(self) -> None:
        assert Severity.ERROR.value == "error"
        assert Severity.WARNING.value == "warning"
        assert Severity.INFO.value == "info"

    def test_quality_category_values(self) -> None:
        assert QualityCategory.IMAGE.value == "image"
        assert QualityCategory.ANNOTATION.value == "annotation"
        assert QualityCategory.CLASS.value == "class"
        assert QualityCategory.GEOMETRY.value == "geometry"
        assert QualityCategory.DUPLICATE.value == "duplicate"
        assert QualityCategory.OUTLIER.value == "outlier"
        assert QualityCategory.IMBALANCE.value == "imbalance"
        assert QualityCategory.COVERAGE.value == "coverage"
        assert QualityCategory.CONSISTENCY.value == "consistency"
        assert QualityCategory.INTEGRITY.value == "integrity"

    def test_letter_grade_values(self) -> None:
        assert LetterGrade.A.value == "A"
        assert LetterGrade.B.value == "B"
        assert LetterGrade.C.value == "C"
        assert LetterGrade.D.value == "D"
        assert LetterGrade.F.value == "F"


class TestQualityIssue:
    def test_create(self) -> None:
        issue = QualityIssue(
            severity=Severity.ERROR,
            category=QualityCategory.IMAGE,
            message="Test issue",
            location="img001",
            suggestion="Fix it",
            details={"key": "value"},
        )
        assert issue.severity == Severity.ERROR
        assert issue.category == QualityCategory.IMAGE
        assert issue.message == "Test issue"
        assert issue.location == "img001"
        assert issue.suggestion == "Fix it"
        assert issue.details == {"key": "value"}

    def test_defaults(self) -> None:
        issue = QualityIssue(
            severity=Severity.INFO,
            category=QualityCategory.CLASS,
            message="Info",
        )
        assert issue.location == ""
        assert issue.suggestion is None
        assert issue.details == {}

    def test_frozen(self) -> None:
        issue = QualityIssue(severity=Severity.INFO, category=QualityCategory.CLASS, message="x")
        with pytest.raises(ValidationError):
            issue.message = "changed"

    def test_message_required(self) -> None:
        with pytest.raises(ValidationError):
            QualityIssue(severity=Severity.INFO, category=QualityCategory.CLASS, message="")


class TestImageValidationResult:
    def test_defaults(self) -> None:
        result = ImageValidationResult()
        assert result.total_images == 0
        assert result.issues == ()
        assert result.passed is True
        assert result.corrupt_count == 0

    def test_custom(self) -> None:
        issue = QualityIssue(severity=Severity.ERROR, category=QualityCategory.IMAGE, message="bad")
        result = ImageValidationResult(
            total_images=10,
            issues=(issue,),
            corrupt_count=2,
            passed=False,
        )
        assert result.total_images == 10
        assert len(result.issues) == 1
        assert result.corrupt_count == 2
        assert result.passed is False

    def test_frozen(self) -> None:
        result = ImageValidationResult()
        with pytest.raises(ValidationError):
            result.passed = False


class TestAnnotationValidationResult:
    def test_defaults(self) -> None:
        result = AnnotationValidationResult()
        assert result.total_annotations == 0
        assert result.passed is True

    def test_with_issues(self) -> None:
        issues = (
            QualityIssue(
                severity=Severity.ERROR, category=QualityCategory.ANNOTATION, message="neg"
            ),
        )
        result = AnnotationValidationResult(
            total_annotations=100,
            negative_coordinate_count=5,
            issues=issues,
            passed=False,
        )
        assert result.total_annotations == 100
        assert result.negative_coordinate_count == 5
        assert result.passed is False


class TestClassValidationResult:
    def test_defaults(self) -> None:
        result = ClassValidationResult()
        assert result.total_classes == 0
        assert result.imbalance_ratio == 1.0

    def test_with_distribution(self) -> None:
        result = ClassValidationResult(
            total_classes=3,
            class_distribution={"car": 100, "person": 50, "truck": 10},
            imbalance_ratio=10.0,
            rare_class_count=1,
        )
        assert result.total_classes == 3
        assert result.class_distribution["car"] == 100
        assert result.imbalance_ratio == 10.0


class TestGeometryValidationResult:
    def test_defaults(self) -> None:
        result = GeometryValidationResult()
        assert result.total_geometries == 0

    def test_counts(self) -> None:
        result = GeometryValidationResult(
            total_geometries=50,
            invalid_bbox_count=3,
            invalid_polygon_count=1,
            passed=False,
        )
        assert result.total_geometries == 50
        assert result.invalid_bbox_count == 3
        assert result.passed is False


class TestDuplicateDetectionResult:
    def test_defaults(self) -> None:
        result = DuplicateDetectionResult()
        assert result.total_duplicate_images == 0
        assert result.passed is True

    def test_with_duplicates(self) -> None:
        result = DuplicateDetectionResult(
            duplicate_image_pairs=(("img001", "img002"),),
            total_duplicate_images=1,
            passed=False,
        )
        assert len(result.duplicate_image_pairs) == 1
        assert result.total_duplicate_images == 1


class TestOutlierDetectionResult:
    def test_defaults(self) -> None:
        result = OutlierDetectionResult()
        assert result.total_outliers == 0

    def test_with_outliers(self) -> None:
        result = OutlierDetectionResult(
            tiny_objects=({"ann_id": "ann001", "area": 2.0},),
            total_outliers=1,
        )
        assert len(result.tiny_objects) == 1


class TestImbalanceAnalysisResult:
    def test_defaults(self) -> None:
        result = ImbalanceAnalysisResult()
        assert result.num_classes == 0
        assert result.imbalance_ratio == 1.0

    def test_with_data(self) -> None:
        result = ImbalanceAnalysisResult(
            class_distribution={"car": 100, "person": 10},
            total_samples=110,
            num_classes=2,
            imbalance_ratio=10.0,
        )
        assert result.total_samples == 110
        assert result.imbalance_ratio == 10.0


class TestCoverageAnalysisResult:
    def test_defaults(self) -> None:
        result = CoverageAnalysisResult()
        assert result.ontology_coverage_pct == 0.0

    def test_with_coverage(self) -> None:
        result = CoverageAnalysisResult(
            ontology_coverage_pct=75.0,
            covered_classes=3,
            total_ontology_classes=4,
        )
        assert result.ontology_coverage_pct == 75.0
        assert result.covered_classes == 3


class TestConsistencyCheckResult:
    def test_defaults(self) -> None:
        result = ConsistencyCheckResult()
        assert result.passed is True
        assert result.metadata_consistent is True

    def test_inconsistent(self) -> None:
        result = ConsistencyCheckResult(
            metadata_consistent=False,
            split_consistent=False,
            passed=False,
        )
        assert result.metadata_consistent is False
        assert result.passed is False


class TestIntegrityCheckResult:
    def test_defaults(self) -> None:
        result = IntegrityCheckResult()
        assert result.passed is True
        assert result.all_files_present is True

    def test_missing_files(self) -> None:
        result = IntegrityCheckResult(
            all_files_present=False,
            missing_files=("img001.jpg",),
            passed=False,
        )
        assert result.all_files_present is False
        assert result.missing_files == ("img001.jpg",)


class TestScoreBreakdown:
    def test_defaults(self) -> None:
        bd = ScoreBreakdown()
        assert bd.image_quality == 0.0
        assert bd.annotation_quality == 0.0

    def test_custom(self) -> None:
        bd = ScoreBreakdown(
            image_quality=85.0,
            annotation_quality=90.0,
            ontology_coverage=70.0,
        )
        assert bd.image_quality == 85.0
        assert bd.annotation_quality == 90.0

    def test_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ScoreBreakdown(image_quality=150.0)
        with pytest.raises(ValidationError):
            ScoreBreakdown(image_quality=-1.0)


class TestDatasetScore:
    def test_defaults(self) -> None:
        score = DatasetScore()
        assert score.overall == 0.0
        assert score.letter_grade == LetterGrade.F
        assert score.production_ready is False

    def test_good_score(self) -> None:
        score = DatasetScore(
            overall=88.5,
            letter_grade=LetterGrade.B,
            production_ready=True,
            breakdown=ScoreBreakdown(
                image_quality=90.0,
                annotation_quality=85.0,
                geometry_quality=80.0,
                ontology_coverage=90.0,
                balance=85.0,
                integrity=90.0,
                consistency=95.0,
            ),
        )
        assert score.overall == 88.5
        assert score.letter_grade == LetterGrade.B
        assert score.production_ready is True

    def test_frozen(self) -> None:
        score = DatasetScore()
        with pytest.raises(ValidationError):
            score.overall = 100.0


class TestQualityReport:
    def test_minimal(self) -> None:
        report = QualityReport(dataset_name="test")
        assert report.dataset_name == "test"
        assert report.dataset_version == "1.0.0"
        assert report.pipeline_version == "1.0.0"
        assert report.error_count == 0

    def test_with_issues(self) -> None:
        issues = (
            QualityIssue(severity=Severity.ERROR, category=QualityCategory.IMAGE, message="err"),
        )
        report = QualityReport(
            dataset_name="test",
            dataset_version="2.0.0",
            all_issues=issues,
            error_count=1,
            warning_count=2,
            info_count=3,
            summary="Test summary",
        )
        assert len(report.all_issues) == 1
        assert report.error_count == 1
        assert report.summary == "Test summary"

    def test_frozen(self) -> None:
        report = QualityReport(dataset_name="test")
        with pytest.raises(ValidationError):
            report.dataset_name = "changed"

    def test_name_required(self) -> None:
        with pytest.raises(ValidationError):
            QualityReport(dataset_name="")

    def test_severity_comparison(self) -> None:
        assert Severity.ERROR != Severity.WARNING  # type: ignore[comparison-overlap]
        assert Severity.WARNING != Severity.INFO  # type: ignore[comparison-overlap]

    def test_quality_category_all(self) -> None:
        categories = {c.value for c in QualityCategory}
        expected = {
            "image",
            "annotation",
            "class",
            "geometry",
            "duplicate",
            "outlier",
            "imbalance",
            "coverage",
            "consistency",
            "integrity",
        }
        assert categories == expected

    def test_quality_issue_with_details(self) -> None:
        issue = QualityIssue(
            severity=Severity.WARNING,
            category=QualityCategory.GEOMETRY,
            message="test",
            details={"count": 5, "items": ["a", "b"]},
        )
        assert issue.details["count"] == 5

    def test_score_breakdown_clamps(self) -> None:
        with pytest.raises(ValidationError):
            ScoreBreakdown(consistency=-5.0)

    def test_score_breakdown_clamps_high(self) -> None:
        with pytest.raises(ValidationError):
            ScoreBreakdown(balance=101.0)

    def test_duplicate_detection_empty_lists(self) -> None:
        result = DuplicateDetectionResult()
        assert result.duplicate_image_pairs == ()
        assert result.duplicate_annotation_pairs == ()

    def test_outlier_detection_empty_lists(self) -> None:
        result = OutlierDetectionResult()
        assert result.extreme_aspect_ratios == ()
        assert result.tiny_objects == ()
        assert result.huge_objects == ()
        assert result.suspicious_annotations == ()

    def test_imbalance_analysis_empty(self) -> None:
        result = ImbalanceAnalysisResult()
        assert result.sorted_distribution == ()
        assert result.recommended_augmentations == {}

    def test_quality_report_with_none_results(self) -> None:
        report = QualityReport(dataset_name="test")
        assert report.image_validation is None
        assert report.annotation_validation is None
        assert report.consistency_check is None

    def test_quality_issue_severity_ordering(self) -> None:
        e = QualityIssue(severity=Severity.ERROR, category=QualityCategory.IMAGE, message="e")
        w = QualityIssue(severity=Severity.WARNING, category=QualityCategory.IMAGE, message="w")
        i = QualityIssue(severity=Severity.INFO, category=QualityCategory.IMAGE, message="i")
        assert e.severity != w.severity
        assert w.severity != i.severity
