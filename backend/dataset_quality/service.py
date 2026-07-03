from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.dataset_report import ReportGenerator
from backend.dataset_quality.models import (
    DatasetScore,
    QualityReport,
)
from backend.dataset_quality.quality_pipeline import QualityPipeline


class DatasetQualityService:
    def __init__(
        self,
        config: DatasetQualityConfig | None = None,
        pipeline: QualityPipeline | None = None,
    ):
        self._config = config or dataset_quality_config
        self._pipeline = pipeline or QualityPipeline(self._config)

    async def run_pipeline(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> QualityReport:
        return await self._pipeline.run(dataset, image_dir, ontology_classes)

    async def quality_score(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> DatasetScore:
        report = await self.run_pipeline(dataset, image_dir, ontology_classes)
        return report.overall_score

    async def generate_report_markdown(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> str:
        report = await self.run_pipeline(dataset, image_dir, ontology_classes)
        generator = ReportGenerator(self._config)
        return await generator.to_markdown(report)

    async def generate_report_json(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> dict[str, object]:
        report = await self.run_pipeline(dataset, image_dir, ontology_classes)
        generator = ReportGenerator(self._config)
        return await generator.to_json(report)
