from __future__ import annotations

import os
import tempfile

import pytest

from backend.dataset_conversion.dataset_loader import DatasetLoader
from backend.dataset_conversion.exceptions import LoadError
from backend.dataset_conversion.models import DatasetFormat


class TestDatasetLoader:
    @pytest.fixture
    def loader(self) -> DatasetLoader:
        return DatasetLoader()

    @pytest.mark.asyncio
    async def test_supported_formats(
        self,
        loader: DatasetLoader,
    ) -> None:
        formats = await loader.supported_formats()
        assert "coco_json" in formats
        assert "yolo_txt" in formats
        assert "pascal_voc" in formats

    @pytest.mark.asyncio
    async def test_load_coco_json(
        self,
        loader: DatasetLoader,
        coco_json_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(coco_json_data)
            f.flush()
            json_path = f.name
        try:
            result = await loader.load(json_path, "coco_json")
            assert result.image_count == 1
            assert result.annotation_count == 1
            assert result.category_count == 1
            assert result.dataset_format == "coco_json"
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_load_coco_missing_file(
        self,
        loader: DatasetLoader,
    ) -> None:
        with pytest.raises(LoadError):
            await loader.load("/nonexistent/path.json", "coco_json")

    @pytest.mark.asyncio
    async def test_load_invalid_json(
        self,
        loader: DatasetLoader,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            json_path = f.name
        try:
            with pytest.raises(LoadError):
                await loader.load(json_path, "coco_json")
        finally:
            os.unlink(json_path)

    @pytest.mark.asyncio
    async def test_load_unsupported_format(
        self,
        loader: DatasetLoader,
    ) -> None:
        with pytest.raises(LoadError):
            await loader.load("/path", "unknown_format")

    @pytest.mark.asyncio
    async def test_yolo_loader_with_image_dir(
        self,
        loader: DatasetLoader,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            labels_dir = os.path.join(tmpdir, "labels")
            images_dir = os.path.join(tmpdir, "images")
            os.makedirs(labels_dir)
            os.makedirs(images_dir)

            label_file = os.path.join(labels_dir, "img001.txt")
            with open(label_file, "w") as f:
                f.write("0 0.5 0.5 0.3 0.4\n1 0.2 0.3 0.1 0.2\n")

            img_file = os.path.join(images_dir, "img001.jpg")
            with open(img_file, "w") as f:
                f.write("fake image data")

            result = await loader.load(
                labels_dir,
                "yolo_txt",
                data_dir=labels_dir,
                image_dir=images_dir,
            )
            assert result.annotation_count >= 2

    @pytest.mark.asyncio
    async def test_pascal_voc_loader(
        self,
        loader: DatasetLoader,
        pascal_voc_xml_data: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = os.path.join(tmpdir, "img001.xml")
            with open(xml_path, "w") as f:
                f.write(pascal_voc_xml_data)

            result = await loader.load(tmpdir, "pascal_voc")
            assert result.annotation_count == 2
            assert result.category_count == 2

    @pytest.mark.asyncio
    async def test_pascal_voc_single_file(
        self,
        loader: DatasetLoader,
        pascal_voc_xml_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write(pascal_voc_xml_data)
            xml_path = f.name
        try:
            result = await loader.load(xml_path, "pascal_voc")
            assert result.annotation_count == 2
        finally:
            os.unlink(xml_path)

    @pytest.mark.asyncio
    async def test_open_images_csv_loader(
        self,
        loader: DatasetLoader,
        open_images_csv_data: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(open_images_csv_data)
            csv_path = f.name
        try:
            result = await loader.load(csv_path, "open_images_csv")
            assert result.annotation_count == 2
            assert result.image_count == 2
        finally:
            os.unlink(csv_path)

    @pytest.mark.asyncio
    async def test_resolve_format(self, loader: DatasetLoader) -> None:
        fmt = loader._resolve_format("coco_json")
        assert fmt == DatasetFormat.COCO_JSON

    def test_resolve_format_invalid(self, loader: DatasetLoader) -> None:
        with pytest.raises(LoadError):
            loader._resolve_format("nonexistent_format")

    def test_find_image_with_valid_extensions(self, loader: DatasetLoader) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            for ext in [".jpg", ".png", ".jpeg"]:
                path = os.path.join(tmpdir, f"test{ext}")
                with open(path, "w") as f:
                    f.write("data")
                result = loader._find_image("test", tmpdir)
                assert result is not None

    def test_find_image_nonexistent(self, loader: DatasetLoader) -> None:
        result = loader._find_image("nonexistent", "/tmp")
        assert result is None

    @pytest.mark.asyncio
    async def test_yolo_loader_no_images(self, loader: DatasetLoader) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "empty.txt"), "w") as f:
                f.write("")
            result = await loader.load(tmpdir, "yolo_txt", image_dir=tmpdir)
            assert result.image_count == 0
            assert result.annotation_count == 0

    @pytest.mark.asyncio
    async def test_open_images_csv_no_class_names(
        self,
        loader: DatasetLoader,
        open_images_csv_data: str,
    ) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(open_images_csv_data)
            csv_path = f.name
        try:
            result = await loader.load(csv_path, "open_images_csv")
            assert result.category_count == 2
        finally:
            os.unlink(csv_path)

    @pytest.mark.asyncio
    async def test_load_canonical_missing_path(self, loader: DatasetLoader) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(LoadError):
                await loader.load(tmpdir, "canonical")

    @pytest.mark.asyncio
    async def test_pascal_voc_invalid_xml(self, loader: DatasetLoader) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False) as f:
            f.write("not valid xml")
            xml_path = f.name
        try:
            with pytest.raises(LoadError):
                await loader.load(xml_path, "pascal_voc")
        finally:
            os.unlink(xml_path)
