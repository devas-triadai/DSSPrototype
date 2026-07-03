from __future__ import annotations

import pytest

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.imbalance_analyzer import ImbalanceAnalyzer


class TestImbalanceAnalyzer:
    @pytest.fixture
    def analyzer(self) -> ImbalanceAnalyzer:
        return ImbalanceAnalyzer()

    @pytest.mark.asyncio
    async def test_balanced_dataset(
        self, analyzer: ImbalanceAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset)
        assert result.passed is True
        assert result.total_samples == sample_dataset.annotation_count
        assert result.num_classes >= 1

    @pytest.mark.asyncio
    async def test_class_distribution(
        self, analyzer: ImbalanceAnalyzer, sample_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_dataset)
        assert len(result.class_distribution) >= 2
        first = result.sorted_distribution[0]["count"]
        last = result.sorted_distribution[-1]["count"]
        assert isinstance(first, int) and isinstance(last, int)
        assert first >= last

    @pytest.mark.asyncio
    async def test_imbalance_detected(self, analyzer: ImbalanceAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id=f"car_{i}",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(100)
        ] + [
            CanonicalAnnotation(
                id=f"person_{i}",
                image_id="img001",
                canonical_label="person",
                canonical_name="person",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(1)
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="imbalanced",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=2,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        assert result.imbalance_ratio >= 50.0
        assert len(result.minority_classes) >= 1

    @pytest.mark.asyncio
    async def test_long_tail_ratio(self, analyzer: ImbalanceAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id=f"ann_{i}",
                image_id="img001",
                canonical_label="class_a",
                canonical_name="a",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(90)
        ]
        anns += [
            CanonicalAnnotation(
                id=f"ann_b{i}",
                image_id="img001",
                canonical_label="class_b",
                canonical_name="b",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(10)
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="longtail",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=2,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        assert result.long_tail_ratio > 0.5

    @pytest.mark.asyncio
    async def test_recommended_augmentations(self, analyzer: ImbalanceAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id=f"ann_{i}",
                image_id="img001",
                canonical_label="majority",
                canonical_name="majority",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(100)
        ]
        anns += [
            CanonicalAnnotation(
                id=f"ann_min_{i}",
                image_id="img001",
                canonical_label="minority",
                canonical_name="minority",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(2)
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="augment",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=2,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        assert len(result.recommended_augmentations) >= 1
        assert "minority" in result.recommended_augmentations

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self, analyzer: ImbalanceAnalyzer, sample_empty_dataset: CanonicalDataset
    ) -> None:
        result = await analyzer.analyze(sample_empty_dataset)
        assert result.passed is True
        assert result.total_samples == 0

    @pytest.mark.asyncio
    async def test_perfectly_balanced(self, analyzer: ImbalanceAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = []
        for cls_name in ["car", "person", "truck"]:
            anns.extend(
                [
                    CanonicalAnnotation(
                        id=f"{cls_name}_{i}",
                        image_id="img001",
                        canonical_label=cls_name,
                        canonical_name=cls_name,
                        geometry_type=GeometryType.BBOX,
                        x=0.0,
                        y=0.0,
                        width=10.0,
                        height=10.0,
                    )
                    for i in range(50)
                ]
            )
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="balanced",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=3,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        assert result.imbalance_ratio == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_single_class_no_imbalance(self, analyzer: ImbalanceAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id=f"ann_{i}",
                image_id="img001",
                canonical_label="car",
                canonical_name="car",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            )
            for i in range(100)
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="single_cls",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=1,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        assert result.num_classes == 1
        assert result.imbalance_ratio == 1.0

    @pytest.mark.asyncio
    async def test_sorted_distribution_order(self, analyzer: ImbalanceAnalyzer) -> None:
        from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType, ImageInfo

        anns = [
            CanonicalAnnotation(
                id="a",
                image_id="img001",
                canonical_label="rare",
                canonical_name="rare",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
            CanonicalAnnotation(
                id="b",
                image_id="img001",
                canonical_label="common",
                canonical_name="common",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
            CanonicalAnnotation(
                id="c",
                image_id="img001",
                canonical_label="common",
                canonical_name="common",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
            CanonicalAnnotation(
                id="d",
                image_id="img001",
                canonical_label="common",
                canonical_name="common",
                geometry_type=GeometryType.BBOX,
                x=0.0,
                y=0.0,
                width=10.0,
                height=10.0,
            ),
        ]
        imgs = [ImageInfo(id="img001", file_path="img.jpg", width=640, height=480)]
        ds = CanonicalDataset(
            name="sorted_t",
            images=tuple(imgs),
            annotations=tuple(anns),
            image_count=1,
            annotation_count=len(anns),
            class_count=2,
            ontology_version="1.0.0",
            pipeline_version="1.0.0",
        )
        result = await analyzer.analyze(ds)
        first = result.sorted_distribution[0]
        last = result.sorted_distribution[-1]
        first_count = first["count"]
        last_count = last["count"]
        assert isinstance(first_count, int) and isinstance(last_count, int)
        assert first_count >= last_count

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import ImbalanceAnalyzerInterface

        assert issubclass(ImbalanceAnalyzer, ImbalanceAnalyzerInterface)
