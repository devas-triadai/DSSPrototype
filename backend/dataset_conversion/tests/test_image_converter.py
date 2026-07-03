from __future__ import annotations

import os
import tempfile

import pytest

from backend.dataset_conversion.image_converter import ImageConverter
from backend.dataset_conversion.models import ImageInfo


class TestImageConverter:
    @pytest.fixture
    def converter(self) -> ImageConverter:
        return ImageConverter()

    @pytest.mark.asyncio
    async def test_validate_image_valid(
        self,
        converter: ImageConverter,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            img_path = f.name
        try:
            img = ImageInfo(id="test", file_path=img_path, width=640, height=480)
            errors = await converter.validate_image(img)
            assert len(errors) == 0
        finally:
            os.unlink(img_path)

    @pytest.mark.asyncio
    async def test_validate_image_missing_file(
        self,
        converter: ImageConverter,
    ) -> None:
        img = ImageInfo(id="test", file_path="/nonexistent/path.jpg")
        errors = await converter.validate_image(img)
        assert len(errors) >= 1
        assert "not found" in errors[0]

    @pytest.mark.asyncio
    async def test_validate_image_no_path(
        self,
        converter: ImageConverter,
    ) -> None:
        img = ImageInfo.model_construct(id="test", file_path="")
        errors = await converter.validate_image(img)
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_validate_image_exceeds_max_dimensions(
        self,
        converter: ImageConverter,
    ) -> None:
        img = ImageInfo(id="test", file_path="/path.jpg", width=5000, height=5000)
        errors = await converter.validate_image(img)
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_validate_image_unsupported_format(
        self,
        converter: ImageConverter,
    ) -> None:
        img = ImageInfo(id="test", file_path="/path.raw", width=100, height=100)
        errors = await converter.validate_image(img)
        assert len(errors) >= 1

    @pytest.mark.asyncio
    async def test_standardize_metadata(
        self,
        converter: ImageConverter,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake image data")
            img_path = f.name
        try:
            img = ImageInfo(id="test", file_path=img_path, width=640, height=480)
            standardized = await converter.standardize_metadata(img)
            assert standardized.format == "jpg"
            assert standardized.color_space == "rgb"
            assert standardized.file_size_bytes is not None
        finally:
            os.unlink(img_path)

    @pytest.mark.asyncio
    async def test_standardize_metadata_no_file(
        self,
        converter: ImageConverter,
    ) -> None:
        img = ImageInfo(id="test", file_path="/nonexistent.jpg")
        standardized = await converter.standardize_metadata(img)
        assert standardized.file_size_bytes is None
        assert standardized.color_space == "rgb"

    @pytest.mark.asyncio
    async def test_standardize_preserves_metadata_dict(
        self,
        converter: ImageConverter,
    ) -> None:
        img = ImageInfo(
            id="test",
            file_path="/path.jpg",
            metadata={"source": "coco"},
        )
        standardized = await converter.standardize_metadata(img)
        assert standardized.metadata == {"source": "coco"}
