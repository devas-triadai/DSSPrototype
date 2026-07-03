from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.quality_pipeline import QualityPipeline


class TestQualityPipeline:
    @pytest.fixture
    def pipeline(self) -> QualityPipeline:
        return QualityPipeline()

    @pytest.mark.asyncio
    async def test_run_pipeline(
        self, pipeline: QualityPipeline, sample_dataset: CanonicalDataset
    ) -> None:
        report = await pipeline.run(sample_dataset)
        assert report.dataset_name == "test_dataset"
        assert report.overall_score.overall > 0
        assert report.image_validation is not None
        assert report.annotation_validation is not None
        assert report.class_validation is not None
        assert report.geometry_validation is not None
        assert report.duplicate_detection is not None
        assert report.outlier_detection is not None
        assert report.imbalance_analysis is not None
        assert report.coverage_analysis is not None
        assert report.consistency_check is not None
        assert report.integrity_check is not None

    @pytest.mark.asyncio
    async def test_run_empty_dataset(
        self, pipeline: QualityPipeline, sample_empty_dataset: CanonicalDataset
    ) -> None:
        report = await pipeline.run(sample_empty_dataset)
        assert report.dataset_name == "empty_dataset"
        assert report.overall_score.overall >= 0
        assert report.image_validation is not None

    @pytest.mark.asyncio
    async def test_run_with_ontology(
        self,
        pipeline: QualityPipeline,
        sample_dataset: CanonicalDataset,
        ontology_classes: list[str],
    ) -> None:
        report = await pipeline.run(sample_dataset, ontology_classes=ontology_classes)
        assert report.class_validation is not None
        assert report.coverage_analysis is not None
        assert report.coverage_analysis.total_ontology_classes == len(ontology_classes)

    @pytest.mark.asyncio
    async def test_pipeline_produces_report(
        self, pipeline: QualityPipeline, sample_dataset: CanonicalDataset
    ) -> None:
        report = await pipeline.run(sample_dataset)
        assert report.all_issues is not None
        assert report.error_count >= 0
        assert report.warning_count >= 0
        assert report.info_count >= 0
        assert report.summary != ""

    @pytest.mark.asyncio
    async def test_run_with_injected_validators(self, sample_dataset: CanonicalDataset) -> None:
        from backend.dataset_quality.annotation_validator import AnnotationValidator
        from backend.dataset_quality.image_validator import ImageValidator

        pipeline = QualityPipeline(
            image_validator=ImageValidator(),
            annotation_validator=AnnotationValidator(),
        )
        report = await pipeline.run(sample_dataset)
        assert report.image_validation is not None
        assert report.annotation_validation is not None

    @pytest.mark.asyncio
    async def test_pipeline_no_errors_on_valid_dataset(
        self, pipeline: QualityPipeline, sample_dataset: CanonicalDataset
    ) -> None:
        report = await pipeline.run(sample_dataset)
        assert report.annotation_validation is not None
        assert report.annotation_validation.passed is True
        assert report.class_validation is not None
        assert report.class_validation.passed is True
        assert report.geometry_validation is not None
        assert report.geometry_validation.passed is True

    @pytest.mark.asyncio
    async def test_pipeline_with_image_dir(
        self, pipeline: QualityPipeline, sample_dataset: CanonicalDataset
    ) -> None:
        report = await pipeline.run(sample_dataset, image_dir="/tmp")
        assert report.integrity_check is not None

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import QualityPipelineInterface

        assert issubclass(QualityPipeline, QualityPipelineInterface)
