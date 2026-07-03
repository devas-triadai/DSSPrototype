from __future__ import annotations

import os
import tempfile

import pytest

from backend.dataset_conversion.manifest_builder import ManifestBuilder
from backend.dataset_conversion.models import (
    CanonicalDataset,
    DatasetStatistics,
    ImageInfo,
)


class TestManifestBuilder:
    @pytest.fixture
    def builder(self) -> ManifestBuilder:
        return ManifestBuilder()

    @pytest.mark.asyncio
    async def test_build_manifest(
        self,
        builder: ManifestBuilder,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        manifest = await builder.build(sample_canonical_dataset)
        assert manifest.dataset_version == "1.0.0"
        assert manifest.ontology_version == "1.0.0"
        assert manifest.pipeline_version == "1.0.0"
        assert manifest.manifest_id is not None
        assert manifest.source_datasets == ("coco",)

    @pytest.mark.asyncio
    async def test_build_with_statistics(
        self,
        builder: ManifestBuilder,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        stats = DatasetStatistics(
            total_images=10,
            total_annotations=50,
            total_classes=3,
            images_with_annotations=8,
            images_without_annotations=2,
            avg_annotations_per_image=5.0,
            min_annotations_per_image=0,
            max_annotations_per_image=15,
            avg_image_width=640.0,
            avg_image_height=480.0,
            avg_bbox_width=100.0,
            avg_bbox_height=150.0,
        )
        manifest = await builder.build(sample_canonical_dataset, stats)
        assert manifest.statistics is not None
        assert manifest.statistics.total_images == 10

    @pytest.mark.asyncio
    async def test_build_without_statistics(
        self,
        builder: ManifestBuilder,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        manifest = await builder.build(sample_canonical_dataset)
        assert manifest.statistics is None

    @pytest.mark.asyncio
    async def test_checksums_for_existing_files(
        self,
        builder: ManifestBuilder,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            img_path = f.name
        try:
            img = ImageInfo(id="img001", file_path=img_path, width=100, height=100)
            ds = CanonicalDataset(
                name="checksum_test",
                images=(img,),
                image_count=1,
                annotation_count=0,
                class_count=0,
            )
            manifest = await builder.build(ds)
            assert img_path in manifest.checksums
            assert len(manifest.checksums[img_path]) == 64
        finally:
            os.unlink(img_path)

    @pytest.mark.asyncio
    async def test_checksums_for_nonexistent_files(
        self,
        builder: ManifestBuilder,
    ) -> None:
        img = ImageInfo(id="img001", file_path="/nonexistent/path.jpg")
        ds = CanonicalDataset(
            name="no_file",
            images=(img,),
            image_count=1,
            annotation_count=0,
            class_count=0,
        )
        manifest = await builder.build(ds)
        assert "/nonexistent/path.jpg" not in manifest.checksums

    @pytest.mark.asyncio
    async def test_manifest_metadata(
        self,
        builder: ManifestBuilder,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        manifest = await builder.build(sample_canonical_dataset)
        assert manifest.metadata["dataset_name"] == "test_dataset"
        assert manifest.metadata["image_count"] == "10"
