from __future__ import annotations

import pytest

from backend.dataset_quality.dataset_scorer import DatasetScorer
from backend.dataset_quality.models import (
    AnnotationValidationResult,
    ClassValidationResult,
    ConsistencyCheckResult,
    CoverageAnalysisResult,
    GeometryValidationResult,
    ImageValidationResult,
    ImbalanceAnalysisResult,
    IntegrityCheckResult,
    LetterGrade,
)


class TestDatasetScorer:
    @pytest.fixture
    def scorer(self) -> DatasetScorer:
        return DatasetScorer()

    @pytest.mark.asyncio
    async def test_perfect_score(self, scorer: DatasetScorer) -> None:
        score = await scorer.score()
        assert score.overall == 100.0
        assert score.letter_grade == LetterGrade.A
        assert score.production_ready is True

    @pytest.mark.asyncio
    async def test_image_quality_scoring(self, scorer: DatasetScorer) -> None:
        img_result = ImageValidationResult(
            total_images=10,
            corrupt_count=5,
        )
        score = await scorer.score(image_result=img_result)
        assert score.breakdown.image_quality < 100.0

    @pytest.mark.asyncio
    async def test_annotation_quality_scoring(self, scorer: DatasetScorer) -> None:
        ann_result = AnnotationValidationResult(
            total_annotations=100,
            negative_coordinate_count=10,
            zero_area_count=5,
        )
        score = await scorer.score(annotation_result=ann_result)
        assert score.breakdown.annotation_quality < 100.0

    @pytest.mark.asyncio
    async def test_geometry_quality_scoring(self, scorer: DatasetScorer) -> None:
        geo_result = GeometryValidationResult(
            total_geometries=100,
            invalid_bbox_count=10,
        )
        score = await scorer.score(geometry_result=geo_result)
        assert score.breakdown.geometry_quality < 100.0

    @pytest.mark.asyncio
    async def test_coverage_scoring(self, scorer: DatasetScorer) -> None:
        cov_result = CoverageAnalysisResult(
            ontology_coverage_pct=80.0,
        )
        score = await scorer.score(coverage_result=cov_result)
        assert score.breakdown.ontology_coverage == 80.0

    @pytest.mark.asyncio
    async def test_balance_scoring_imbalanced(self, scorer: DatasetScorer) -> None:
        imb_result = ImbalanceAnalysisResult(
            class_distribution={"car": 100, "person": 1},
            num_classes=2,
            imbalance_ratio=100.0,
        )
        score = await scorer.score(imbalance_result=imb_result)
        assert score.breakdown.balance < 100.0

    @pytest.mark.asyncio
    async def test_integrity_scoring(self, scorer: DatasetScorer) -> None:
        int_result = IntegrityCheckResult(
            checksums_valid=False,
            manifest_valid=False,
            all_files_present=True,
            no_broken_references=True,
            version_valid=True,
            passed=False,
        )
        score = await scorer.score(integrity_result=int_result)
        assert score.breakdown.integrity == 60.0

    @pytest.mark.asyncio
    async def test_consistency_scoring(self, scorer: DatasetScorer) -> None:
        con_result = ConsistencyCheckResult(
            metadata_consistent=True,
            split_consistent=True,
            ontology_consistent=True,
            annotation_consistent=False,
            version_consistent=True,
            passed=False,
        )
        score = await scorer.score(consistency_result=con_result)
        assert score.breakdown.consistency == 80.0

    @pytest.mark.asyncio
    async def test_production_ready_threshold(self, scorer: DatasetScorer) -> None:
        score = await scorer.score()
        assert score.production_ready is True
        assert score.letter_grade == LetterGrade.A

    @pytest.mark.asyncio
    async def test_not_production_ready(self, scorer: DatasetScorer) -> None:
        ann_result = AnnotationValidationResult(
            total_annotations=10,
            negative_coordinate_count=5,
            zero_area_count=3,
            passed=False,
        )
        score = await scorer.score(annotation_result=ann_result)
        assert not score.production_ready

    @pytest.mark.asyncio
    async def test_letter_grade_a(self, scorer: DatasetScorer) -> None:
        assert scorer._letter_grade(95.0) == LetterGrade.A

    def test_letter_grade_b(self, scorer: DatasetScorer) -> None:
        assert scorer._letter_grade(80.0) == LetterGrade.B

    def test_letter_grade_c(self, scorer: DatasetScorer) -> None:
        assert scorer._letter_grade(65.0) == LetterGrade.C

    def test_letter_grade_d(self, scorer: DatasetScorer) -> None:
        assert scorer._letter_grade(45.0) == LetterGrade.D

    def test_letter_grade_f(self, scorer: DatasetScorer) -> None:
        assert scorer._letter_grade(20.0) == LetterGrade.F

    @pytest.mark.asyncio
    async def test_score_annotation_with_missing(self, scorer: DatasetScorer) -> None:
        ann_result = AnnotationValidationResult(
            total_annotations=50,
            total_images=60,
            missing_annotation_count=10,
        )
        score = await scorer.score(annotation_result=ann_result)
        assert score.breakdown.annotation_quality < 100.0

    @pytest.mark.asyncio
    async def test_score_geometry_empty(self, scorer: DatasetScorer) -> None:
        geo_result = GeometryValidationResult(total_geometries=0)
        score = await scorer.score(geometry_result=geo_result)
        assert score.breakdown.geometry_quality == 100.0

    @pytest.mark.asyncio
    async def test_score_balance_with_class(self, scorer: DatasetScorer) -> None:
        cls_result = ClassValidationResult(
            class_distribution={"car": 100, "person": 1},
            rare_class_count=1,
        )
        score = await scorer.score(class_result=cls_result)
        assert score.breakdown.balance < 100.0

    @pytest.mark.asyncio
    async def test_all_subscores_low(self, scorer: DatasetScorer) -> None:
        img = ImageValidationResult(total_images=10, corrupt_count=10)
        ann = AnnotationValidationResult(
            total_annotations=10, negative_coordinate_count=10, zero_area_count=5
        )
        geo = GeometryValidationResult(total_geometries=10, invalid_bbox_count=10)
        cov = CoverageAnalysisResult(ontology_coverage_pct=0.0)
        imb = ImbalanceAnalysisResult(
            class_distribution={"a": 100, "b": 1}, num_classes=2, imbalance_ratio=100.0
        )
        integ = IntegrityCheckResult(
            checksums_valid=False,
            manifest_valid=False,
            all_files_present=False,
            no_broken_references=False,
            version_valid=False,
            passed=False,
        )
        cons = ConsistencyCheckResult(
            metadata_consistent=False,
            split_consistent=False,
            ontology_consistent=False,
            annotation_consistent=False,
            version_consistent=False,
            passed=False,
        )
        score = await scorer.score(
            image_result=img,
            annotation_result=ann,
            geometry_result=geo,
            coverage_result=cov,
            imbalance_result=imb,
            integrity_result=integ,
            consistency_result=cons,
        )
        assert score.overall < 50.0
        assert score.letter_grade in (LetterGrade.D, LetterGrade.F)

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import DatasetScorerInterface

        assert issubclass(DatasetScorer, DatasetScorerInterface)
