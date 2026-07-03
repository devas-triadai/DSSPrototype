from __future__ import annotations

import os
import tempfile

import pytest

from backend.dataset_conversion.models import (
    CanonicalDataset,
    MergeConfig,
    SplitConfig,
)
from backend.dataset_conversion.service import DatasetConversionService


class TestDatasetConversionService:
    @pytest.fixture
    def service(self) -> DatasetConversionService:
        return DatasetConversionService()

    @pytest.mark.asyncio
    async def test_load_dataset(
        self,
        service: DatasetConversionService,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            result = await service.load_dataset(json_path, "coco_json")
            assert result.image_count == 1
            assert result.annotation_count == 1
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_convert_dataset(
        self,
        service: DatasetConversionService,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            load_result = await service.load_dataset(json_path, "coco_json")
            dataset = await service.convert_dataset(load_result, "test_conv")
            assert dataset.name == "test_conv"
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_run_pipeline(
        self,
        service: DatasetConversionService,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await service.run_pipeline(
                    source_path=json_path,
                    source_format="coco_json",
                    dataset_name="pipeline_test",
                    output_path=tmpdir,
                )
                assert result.dataset.name == "pipeline_test"
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_validate_dataset(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        report = await service.validate_dataset(sample_canonical_dataset)
        assert report.valid
        assert report.total_checks == 8

    @pytest.mark.asyncio
    async def test_dataset_statistics(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        stats = await service.dataset_statistics(sample_canonical_dataset)
        assert stats.total_images == 10
        assert stats.total_annotations == 10

    @pytest.mark.asyncio
    async def test_split_dataset(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        result = await service.split_dataset(sample_canonical_dataset)
        assert result.train.image_count > 0
        assert result.val.image_count > 0

    @pytest.mark.asyncio
    async def test_merge_datasets(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        result = await service.merge_datasets([sample_canonical_dataset, sample_canonical_dataset])
        assert result.total_images > 0

    @pytest.mark.asyncio
    async def test_export_dataset(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await service.export_dataset(
                sample_canonical_dataset,
                "coco_json",
                tmpdir,
            )
            assert result.images_exported == 10

    @pytest.mark.asyncio
    async def test_build_manifest(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        manifest = await service.build_manifest(sample_canonical_dataset)
        assert manifest.dataset_version == "1.0.0"
        assert manifest.ontology_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_full_lifecycle(
        self,
        service: DatasetConversionService,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            load_result = await service.load_dataset(json_path, "coco_json")
            dataset = await service.convert_dataset(load_result, "lifecycle_test")
            report = await service.validate_dataset(dataset)
            assert report.valid
            stats = await service.dataset_statistics(dataset)
            assert stats.total_images >= 0
            split = await service.split_dataset(dataset)
            assert split.train.image_count >= 0
            export = await service.export_dataset(dataset, "canonical", os.path.dirname(json_path))
            assert export.images_exported >= 0
            manifest = await service.build_manifest(dataset, stats)
            assert manifest is not None
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_split_with_custom_config(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = SplitConfig(train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=99)
        result = await service.split_dataset(sample_canonical_dataset, config)
        assert result.train.image_count > 0

    @pytest.mark.asyncio
    async def test_merge_with_custom_prefix(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        config = MergeConfig(image_id_prefix="custom", annotation_id_prefix="custom_ann")
        result = await service.merge_datasets([sample_canonical_dataset], config)
        assert result.dataset.name == "custom"

    @pytest.mark.asyncio
    async def test_export_with_unsupported_format(
        self,
        service: DatasetConversionService,
        sample_canonical_dataset: CanonicalDataset,
    ) -> None:
        with pytest.raises(ValueError):
            await service.export_dataset(sample_canonical_dataset, "unsupported", "/tmp")
