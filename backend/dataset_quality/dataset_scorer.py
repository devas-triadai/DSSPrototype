from __future__ import annotations

from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import DatasetScorerInterface
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
    QualityIssue,
    ScoreBreakdown,
)


class DatasetScorer(DatasetScorerInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

    async def score(
        self,
        image_result: ImageValidationResult | None = None,
        annotation_result: AnnotationValidationResult | None = None,
        geometry_result: GeometryValidationResult | None = None,
        class_result: ClassValidationResult | None = None,
        duplicate_result: DuplicateDetectionResult | None = None,
        outlier_result: OutlierDetectionResult | None = None,
        imbalance_result: ImbalanceAnalysisResult | None = None,
        coverage_result: CoverageAnalysisResult | None = None,
        consistency_result: ConsistencyCheckResult | None = None,
        integrity_result: IntegrityCheckResult | None = None,
    ) -> DatasetScore:
        issues: list[QualityIssue] = []

        image_quality = await self._score_image(image_result)
        annotation_quality = await self._score_annotation(annotation_result)
        geometry_quality = await self._score_geometry(geometry_result)
        ontology_coverage = await self._score_coverage(coverage_result)
        balance = await self._score_balance(imbalance_result, class_result)
        integrity = await self._score_integrity(integrity_result)
        consistency = await self._score_consistency(consistency_result)

        weights = {
            "image": 0.10,
            "annotation": 0.25,
            "geometry": 0.15,
            "coverage": 0.15,
            "balance": 0.10,
            "integrity": 0.15,
            "consistency": 0.10,
        }

        overall = (
            image_quality * weights["image"]
            + annotation_quality * weights["annotation"]
            + geometry_quality * weights["geometry"]
            + ontology_coverage * weights["coverage"]
            + balance * weights["balance"]
            + integrity * weights["integrity"]
            + consistency * weights["consistency"]
        )

        letter_grade = self._letter_grade(overall)

        has_errors = (
            (image_result and not image_result.passed)
            or (annotation_result and not annotation_result.passed)
            or (geometry_result and not geometry_result.passed)
            or (class_result and not class_result.passed)
            or (duplicate_result and not duplicate_result.passed)
            or (integrity_result and not integrity_result.passed)
            or (consistency_result and not consistency_result.passed)
        )
        production_ready = overall >= 75.0 and not has_errors

        return DatasetScore(
            overall=round(overall, 1),
            letter_grade=letter_grade,
            production_ready=production_ready,
            breakdown=ScoreBreakdown(
                image_quality=round(image_quality, 1),
                annotation_quality=round(annotation_quality, 1),
                geometry_quality=round(geometry_quality, 1),
                ontology_coverage=round(ontology_coverage, 1),
                balance=round(balance, 1),
                integrity=round(integrity, 1),
                consistency=round(consistency, 1),
            ),
            issues=tuple(issues),
        )

    async def _score_image(self, result: ImageValidationResult | None) -> float:
        if result is None or result.total_images == 0:
            return 100.0
        error_count = result.corrupt_count + result.unreadable_count
        warning_count = (
            result.wrong_format_count
            + result.wrong_color_space_count
            + result.tiny_image_count
            + result.oversized_image_count
            + result.blank_image_count
        )
        score = (
            100.0
            - (error_count / result.total_images) * 100
            - (warning_count / result.total_images) * 25
        )
        return max(0.0, min(100.0, score))

    async def _score_annotation(self, result: AnnotationValidationResult | None) -> float:
        if result is None or result.total_annotations == 0:
            return 100.0
        bad_count = (
            result.negative_coordinate_count + result.zero_area_count + result.broken_obb_count
        )
        warning_count = (
            result.out_of_bounds_count
            + result.invalid_polygon_count
            + result.broken_segmentation_count
            + result.missing_annotation_count
        )
        total = result.total_annotations + max(0, result.missing_annotation_count - 1)
        score = 100.0 - (bad_count / total) * 100 - (warning_count / total) * 20
        return max(0.0, min(100.0, score))

    async def _score_geometry(self, result: GeometryValidationResult | None) -> float:
        if result is None or result.total_geometries == 0:
            return 100.0
        bad_count = (
            result.invalid_bbox_count
            + result.invalid_segmentation_count
            + result.invalid_normalized_coord_count
        )
        score = 100.0 - (bad_count / result.total_geometries) * 100
        return max(0.0, min(100.0, score))

    async def _score_coverage(self, result: CoverageAnalysisResult | None) -> float:
        if result is None:
            return 100.0
        return result.ontology_coverage_pct

    async def _score_balance(
        self,
        imbalance_result: ImbalanceAnalysisResult | None,
        class_result: ClassValidationResult | None,
    ) -> float:
        if imbalance_result is None and class_result is None:
            return 100.0

        score = 100.0
        if imbalance_result and imbalance_result.num_classes > 1:
            ratio = imbalance_result.imbalance_ratio
            if ratio > 1:
                score -= min(50, (ratio - 1) * 5)

        if class_result:
            rare_pct = 0.0
            if class_result.class_distribution:
                total = sum(class_result.class_distribution.values())
                if total > 0:
                    rare_pct = (
                        class_result.rare_class_count / len(class_result.class_distribution)
                        if class_result.class_distribution
                        else 0.0
                    )
                score -= rare_pct * 50

        return max(0.0, min(100.0, score))

    async def _score_integrity(self, result: IntegrityCheckResult | None) -> float:
        if result is None:
            return 100.0
        checks = [
            result.checksums_valid,
            result.manifest_valid,
            result.all_files_present,
            result.no_broken_references,
            result.version_valid,
        ]
        passed = sum(1 for c in checks if c)
        return (passed / len(checks)) * 100

    async def _score_consistency(self, result: ConsistencyCheckResult | None) -> float:
        if result is None:
            return 100.0
        checks = [
            result.metadata_consistent,
            result.split_consistent,
            result.ontology_consistent,
            result.annotation_consistent,
            result.version_consistent,
        ]
        passed = sum(1 for c in checks if c)
        return (passed / len(checks)) * 100

    def _letter_grade(self, score: float) -> LetterGrade:
        if score >= 90.0:
            return LetterGrade.A
        if score >= 75.0:
            return LetterGrade.B
        if score >= 60.0:
            return LetterGrade.C
        if score >= 40.0:
            return LetterGrade.D
        return LetterGrade.F
