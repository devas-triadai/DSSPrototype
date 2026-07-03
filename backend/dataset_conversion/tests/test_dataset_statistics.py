from __future__ import annotations

import pytest

from backend.dataset_conversion.dataset_statistics import DatasetStatisticsGenerator
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    GeometryType,
    ImageInfo,
)


class TestDatasetStatistics:
    @pytest.fixture
    def stats_gen(self) -> DatasetStatisticsGenerator:
        return DatasetStatisticsGenerator()

    @pytest.mark.asyncio
    async def test_compute_basic(
        self,
        stats_gen: DatasetStatisticsGenerator,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        stats = await stats_gen.compute(sample_canonical_dataset)
        assert stats.total_images == 10
        assert stats.total_annotations == 10
        assert stats.total_classes == 2

    @pytest.mark.asyncio
    async def test_empty_dataset(
        self,
        stats_gen: DatasetStatisticsGenerator,
    ) -> None:
        ds = CanonicalDataset(name="empty", image_count=0, annotation_count=0, class_count=0)
        stats = await stats_gen.compute(ds)
        assert stats.total_images == 0
        assert stats.total_annotations == 0
        assert stats.avg_annotations_per_image == 0.0

    @pytest.mark.asyncio
    async def test_class_counts(
        self,
        stats_gen: DatasetStatisticsGenerator,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        stats = await stats_gen.compute(sample_canonical_dataset)
        assert len(stats.classes) == 2
        class_names = [c[0] for c in stats.classes]
        assert "ground_vehicle.car" in class_names
        assert "people.person" in class_names

    @pytest.mark.asyncio
    async def test_class_balance(
        self,
        stats_gen: DatasetStatisticsGenerator,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        stats = await stats_gen.compute(sample_canonical_dataset)
        assert len(stats.class_balance) == 2
        total_proportion = sum(stats.class_balance.values())
        assert abs(total_proportion - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_images_with_and_without_annotations(
        self,
        stats_gen: DatasetStatisticsGenerator,
    ) -> None:
        images = [ImageInfo(id=f"img{i:03d}", file_path=f"/path/{i}.jpg") for i in range(10)]
        anns = [
            CanonicalAnnotation(
                id=f"ann{i:03d}",
                image_id=f"img{i:03d}",
                canonical_label="a.b",
                canonical_name="A",
                geometry_type=GeometryType.BBOX,
                x=0,
                y=0,
                width=10,
                height=10,
            )
            for i in range(5)
        ]
        ds = CanonicalDataset(
            name="partial",
            images=tuple(images),
            annotations=tuple(anns),
            image_count=10,
            annotation_count=5,
            class_count=1,
        )
        stats = await stats_gen.compute(ds)
        assert stats.images_without_annotations == 5

    @pytest.mark.asyncio
    async def test_min_max_annotations(
        self,
        stats_gen: DatasetStatisticsGenerator,
    ) -> None:
        images = [ImageInfo(id=f"img{i:03d}", file_path=f"/path/{i}.jpg") for i in range(3)]
        anns = []
        for i in range(3):
            for j in range(i + 1):
                anns.append(
                    CanonicalAnnotation(
                        id=f"ann{i}_{j}",
                        image_id=f"img{i:03d}",
                        canonical_label="a.b",
                        canonical_name="A",
                        geometry_type=GeometryType.BBOX,
                        x=0,
                        y=0,
                        width=10,
                        height=10,
                    )
                )
        ds = CanonicalDataset(
            name="varying",
            images=tuple(images),
            annotations=tuple(anns),
            image_count=3,
            annotation_count=3,
            class_count=1,
        )
        stats = await stats_gen.compute(ds)
        assert stats.min_annotations_per_image == 1
        assert stats.max_annotations_per_image == 3

    @pytest.mark.asyncio
    async def test_avg_bbox_dimensions(
        self,
        stats_gen: DatasetStatisticsGenerator,
    ) -> None:
        images = [ImageInfo(id=f"img{i}", file_path=f"/path/{i}.jpg") for i in range(5)]
        anns = [
            CanonicalAnnotation(
                id=f"ann{i}",
                image_id=f"img{i}",
                canonical_label="a.b",
                canonical_name="A",
                geometry_type=GeometryType.BBOX,
                x=0,
                y=0,
                width=(i + 1) * 10.0,
                height=(i + 1) * 20.0,
            )
            for i in range(5)
        ]
        ds = CanonicalDataset(
            name="bbox_stats",
            images=tuple(images),
            annotations=tuple(anns),
            image_count=5,
            annotation_count=5,
            class_count=1,
        )
        stats = await stats_gen.compute(ds)
        assert stats.avg_bbox_width == 30.0
        assert stats.avg_bbox_height == 60.0

    @pytest.mark.asyncio
    async def test_coverage_report(
        self,
        stats_gen: DatasetStatisticsGenerator,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        stats = await stats_gen.compute(sample_canonical_dataset)
        assert "coverage_pct" in stats.coverage_report
        assert stats.coverage_report["coverage_pct"] == 100.0
