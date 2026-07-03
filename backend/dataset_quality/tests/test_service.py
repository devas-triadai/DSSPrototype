from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.service import DatasetQualityService


class TestDatasetQualityService:
    @pytest.fixture
    def service(self) -> DatasetQualityService:
        return DatasetQualityService()

    @pytest.mark.asyncio
    async def test_run_pipeline(
        self, service: DatasetQualityService, sample_dataset: CanonicalDataset
    ) -> None:
        report = await service.run_pipeline(sample_dataset)
        assert report.dataset_name == "test_dataset"
        assert report.overall_score is not None

    @pytest.mark.asyncio
    async def test_quality_score(
        self, service: DatasetQualityService, sample_dataset: CanonicalDataset
    ) -> None:
        score = await service.quality_score(sample_dataset)
        assert score.overall > 0
        assert score.letter_grade is not None

    @pytest.mark.asyncio
    async def test_quality_score_empty(
        self, service: DatasetQualityService, sample_empty_dataset: CanonicalDataset
    ) -> None:
        score = await service.quality_score(sample_empty_dataset)
        assert score.overall >= 0

    @pytest.mark.asyncio
    async def test_generate_report_markdown(
        self, service: DatasetQualityService, sample_dataset: CanonicalDataset
    ) -> None:
        md = await service.generate_report_markdown(sample_dataset)
        assert isinstance(md, str)
        assert "# Quality Report:" in md
        assert sample_dataset.name in md

    @pytest.mark.asyncio
    async def test_generate_report_json(
        self, service: DatasetQualityService, sample_dataset: CanonicalDataset
    ) -> None:
        json_data = await service.generate_report_json(sample_dataset)
        assert isinstance(json_data, dict)
        assert json_data["dataset_name"] == sample_dataset.name
        assert "overall_score" in json_data

    @pytest.mark.asyncio
    async def test_with_ontology_classes(
        self,
        service: DatasetQualityService,
        sample_dataset: CanonicalDataset,
        ontology_classes: list[str],
    ) -> None:
        report = await service.run_pipeline(sample_dataset, ontology_classes=ontology_classes)
        assert report.coverage_analysis is not None
        assert report.coverage_analysis.total_ontology_classes == len(ontology_classes)

    @pytest.mark.asyncio
    async def test_empty_dataset_markdown(
        self, service: DatasetQualityService, sample_empty_dataset: CanonicalDataset
    ) -> None:
        md = await service.generate_report_markdown(sample_empty_dataset)
        assert isinstance(md, str)
        assert sample_empty_dataset.name in md

    @pytest.mark.asyncio
    async def test_inject_custom_pipeline(self, sample_dataset: CanonicalDataset) -> None:
        from backend.dataset_quality.quality_pipeline import QualityPipeline

        pipeline = QualityPipeline()
        service = DatasetQualityService(pipeline=pipeline)
        report = await service.run_pipeline(sample_dataset)
        assert report.dataset_name == sample_dataset.name

    @pytest.mark.asyncio
    async def test_quality_score_json_consistency(
        self, service: DatasetQualityService, sample_dataset: CanonicalDataset
    ) -> None:
        score = await service.quality_score(sample_dataset)
        json_data = await service.generate_report_json(sample_dataset)
        overall = json_data["overall_score"]
        assert isinstance(overall, dict) and overall.get("overall") == score.overall
