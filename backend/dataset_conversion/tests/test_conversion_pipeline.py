from __future__ import annotations

import os
import tempfile

import pytest

from backend.dataset_conversion.conversion_pipeline import ConversionPipeline


class TestConversionPipeline:
    @pytest.fixture
    def pipeline(self) -> ConversionPipeline:
        return ConversionPipeline()

    @pytest.mark.asyncio
    async def test_run_coco(
        self,
        pipeline: ConversionPipeline,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await pipeline.run(
                    source_path=json_path,
                    source_format="coco_json",
                    dataset_name="test_run",
                    output_path=tmpdir,
                )
                assert result.dataset.name == "test_run"
                assert result.dataset.image_count >= 0
                assert result.dataset.annotation_count >= 0
                assert result.report.total_images >= 0
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_run_invalid_format(
        self,
        pipeline: ConversionPipeline,
    ) -> None:
        with pytest.raises(Exception):
            await pipeline.run(
                source_path="/nonexistent",
                source_format="unknown",
                dataset_name="fail",
                output_path="/tmp",
            )

    @pytest.mark.asyncio
    async def test_run_pascal_voc(
        self,
        pipeline: ConversionPipeline,
        pascal_voc_xml_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(pascal_voc_xml_data)
            xml_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await pipeline.run(
                    source_path=xml_path,
                    source_format="pascal_voc",
                    dataset_name="voc_test",
                    output_path=tmpdir,
                )
                assert result.dataset.name == "voc_test"
                assert result.report.total_annotations >= 0
        finally:
            os.unlink(xml_path)

    @pytest.mark.asyncio
    async def test_run_with_invalid_image_dir(
        self,
        pipeline: ConversionPipeline,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                result = await pipeline.run(
                    source_path=json_path,
                    source_format="coco_json",
                    dataset_name="test_no_imgs",
                    output_path=tmpdir,
                    data_dir="/nonexistent",
                )
                assert result.dataset.image_count == 1
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_convert_method(
        self,
        pipeline: ConversionPipeline,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            json_path = f.name
        try:
            from backend.dataset_conversion.dataset_loader import DatasetLoader

            loader = DatasetLoader()
            load_result = await loader.load(json_path, "coco_json")
            dataset = await pipeline._convert(load_result, "test_convert")
            assert dataset.name == "test_convert"
        finally:
            os.unlink(json_path)
