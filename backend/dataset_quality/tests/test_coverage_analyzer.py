from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.coverage_analyzer import CoverageAnalyzer


class TestCoverageAnalyzer:
    @pytest.fixture
    def analyzer(self) -> CoverageAnalyzer:
        return CoverageAnalyzer()

    @pytest.mark.asyncio
    async def test_full_coverage(
        self,
        analyzer: CoverageAnalyzer,
        sample_dataset: CanonicalDataset,
        ontology_classes: list[str],
    ) -> None:
        result = await analyzer.analyze(sample_dataset, ontology_classes)
        assert result.total_ontology_classes == len(ontology_classes)
        assert result.covered_classes >= 2

    @pytest.mark.asyncio
    async def test_missing_classes(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        all_classes = ["aerial.drone", "ground_vehicle.bus", "water.boat"]
        result = await analyzer.analyze(sample_dataset, all_classes)
        assert result.covered_classes == 0
        assert len(result.missing_classes) == 3

    @pytest.mark.asyncio
    async def test_no_ontology_provided(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset)
        assert result.passed is True
        assert result.covered_classes >= 2

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, analyzer: CoverageAnalyzer, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_empty_dataset, ["car", "person"])
        assert result.ontology_coverage_pct == 0.0
        assert result.image_coverage_pct == 0.0

    @pytest.mark.asyncio
    async def test_image_coverage(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset)
        assert result.image_coverage_pct > 0

    @pytest.mark.asyncio
    async def test_completeness_pct(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset, ["ground_vehicle.car", "people.person"])
        assert result.completeness_pct > 0

    @pytest.mark.asyncio
    async def test_low_coverage_warning(self, analyzer: CoverageAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="ann001",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="low_cov",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=1,
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        many_classes = [f"class_{i}" for i in range(20)]
        result = await analyzer.analyze(ds, many_classes)
        assert result.ontology_coverage_pct < 50
        assert len(result.issues) >= 1

    @pytest.mark.asyncio
    async def test_object_diversity_nonzero(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset, ["ground_vehicle.car", "people.person"])
        assert result.object_diversity > 0

    @pytest.mark.asyncio
    async def test_scene_diversity(self, analyzer: CoverageAnalyzer) -> None:
        from backend.dataset_conversion.models import ImageInfo

        imgs = [
            ImageInfo(id="img001", file_path="/data/scene1_img001.jpg", width=640, height=480),
            ImageInfo(id="img002", file_path="/data/scene2_img002.jpg", width=640, height=480),
            ImageInfo(id="img003", file_path="/data/scene3_img003.jpg", width=640, height=480),
        ]
        ds = CanonicalDataset(
            name="diverse",
            images=tuple(imgs),
            annotations=(),
            image_count=3,
            annotation_count=0,
            class_count=0,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        assert result.scene_diversity >= 0

    @pytest.mark.asyncio
    async def test_full_coverage_with_mixed(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(
            sample_dataset,
            [
                "ground_vehicle.car",
                "people.person",
                "ground_vehicle.truck",
            ],
        )
        assert result.ontology_coverage_pct == 100.0

    @pytest.mark.asyncio
    async def test_no_images_empty_coverage(
        self, analyzer: CoverageAnalyzer, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_empty_dataset, ["car"])
        assert result.image_coverage_pct == 0.0
        assert result.ontology_coverage_pct == 0.0

    @pytest.mark.asyncio
    async def test_empty_ontology_classes(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset, [])
        assert result.total_ontology_classes > 0
        assert result.ontology_coverage_pct == 100.0

    @pytest.mark.asyncio
    async def test_partial_overlap(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset, ["ground_vehicle.car", "aerial.drone"])
        assert result.covered_classes == 1
        assert len(result.missing_classes) == 1

    @pytest.mark.asyncio
    async def test_no_overlap(
        self, analyzer: CoverageAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset, ["aerial.drone", "water.boat"])
        assert result.covered_classes == 0

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import CoverageAnalyzerInterface

        assert issubclass(CoverageAnalyzer, CoverageAnalyzerInterface)
