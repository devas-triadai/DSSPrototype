from __future__ import annotations

import pytest

from backend.dataset_conversion.dataset_merger import DatasetMerger
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    GeometryType,
    ImageInfo,
    MergeConfig,
)


class TestDatasetMerger:
    @pytest.fixture
    def merger(self) -> DatasetMerger:
        return DatasetMerger()

    @pytest.fixture
    def ds_a(self) -> CanonicalDataset:
        images = (ImageInfo(id="img001", file_path="/a/img001.jpg", width=640, height=480),)
        anns = (
            CanonicalAnnotation(
                id="ann001",
                image_id="img001",
                canonical_label="ground_vehicle.car",
                canonical_name="Car",
                geometry_type=GeometryType.BBOX,
                x=10,
                y=20,
                width=100,
                height=200,
            ),
        )
        return CanonicalDataset(
            name="ds_a",
            images=images,
            annotations=anns,
            image_count=1,
            annotation_count=1,
            class_count=1,
            source_datasets=("ds_a",),
        )

    @pytest.fixture
    def ds_b(self) -> CanonicalDataset:
        images = (ImageInfo(id="img002", file_path="/b/img002.jpg", width=800, height=600),)
        anns = (
            CanonicalAnnotation(
                id="ann002",
                image_id="img002",
                canonical_label="people.person",
                canonical_name="Person",
                geometry_type=GeometryType.BBOX,
                x=50,
                y=60,
                width=80,
                height=160,
            ),
        )
        return CanonicalDataset(
            name="ds_b",
            images=images,
            annotations=anns,
            image_count=1,
            annotation_count=1,
            class_count=1,
            source_datasets=("ds_b",),
        )

    @pytest.fixture
    def ds_overlap(self) -> CanonicalDataset:
        images = (ImageInfo(id="img001", file_path="/c/img001.jpg", width=640, height=480),)
        anns = (
            CanonicalAnnotation(
                id="ann003",
                image_id="img001",
                canonical_label="ground_vehicle.car",
                canonical_name="Car",
                geometry_type=GeometryType.BBOX,
                x=10,
                y=20,
                width=100,
                height=200,
            ),
        )
        return CanonicalDataset(
            name="ds_c",
            images=images,
            annotations=anns,
            image_count=1,
            annotation_count=1,
            class_count=1,
            source_datasets=("ds_c",),
        )

    @pytest.mark.asyncio
    async def test_merge_two_datasets(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
        ds_b: CanonicalDataset,
    ) -> None:
        result = await merger.merge([ds_a, ds_b])
        assert result.total_images == 2
        assert result.total_annotations == 2
        assert len(result.source_datasets) == 2

    @pytest.mark.asyncio
    async def test_merge_deduplicates_images(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
        ds_overlap: CanonicalDataset,
    ) -> None:
        result = await merger.merge([ds_a, ds_overlap])
        assert result.deduplicated_count >= 1

    @pytest.mark.asyncio
    async def test_merge_empty_list(
        self,
        merger: DatasetMerger,
    ) -> None:
        with pytest.raises(ValueError, match="Cannot merge empty"):
            await merger.merge([])

    @pytest.mark.asyncio
    async def test_merge_with_custom_config(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
        ds_b: CanonicalDataset,
    ) -> None:
        config = MergeConfig(
            strategy="sequential",
            image_id_prefix="merged_ds",
            annotation_id_prefix="merged_ann",
        )
        result = await merger.merge([ds_a, ds_b], config)
        assert result.dataset.name == "merged_ds"
        assert result.total_images == 2

    @pytest.mark.asyncio
    async def test_merge_no_dedup(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
        ds_overlap: CanonicalDataset,
    ) -> None:
        config = MergeConfig(deduplicate_images=False)
        result = await merger.merge([ds_a, ds_overlap], config)
        assert result.total_images == 2

    @pytest.mark.asyncio
    async def test_merge_preserves_provenance(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
        ds_b: CanonicalDataset,
    ) -> None:
        result = await merger.merge([ds_a, ds_b])
        assert "ds_a" in result.source_datasets
        assert "ds_b" in result.source_datasets

    @pytest.mark.asyncio
    async def test_merge_single_dataset(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
    ) -> None:
        result = await merger.merge([ds_a])
        assert result.total_images == 1
        assert result.total_annotations == 1

    @pytest.mark.asyncio
    async def test_merge_annotation_metadata_has_merged_from(
        self,
        merger: DatasetMerger,
        ds_a: CanonicalDataset,
        ds_b: CanonicalDataset,
    ) -> None:
        result = await merger.merge([ds_a, ds_b])
        for ann in result.dataset.annotations:
            assert "merged_from" in ann.metadata
