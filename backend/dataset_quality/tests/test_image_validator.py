from __future__ import annotations

from collections.abc import Sequence

import pytest

from backend.dataset_conversion.models import ImageInfo
from backend.dataset_quality.config import DatasetQualityConfig
from backend.dataset_quality.image_validator import ImageValidator


class TestImageValidator:
    @pytest.fixture
    def validator(self) -> ImageValidator:
        return ImageValidator()

    @pytest.fixture
    def strict_validator(self) -> ImageValidator:
        cfg = DatasetQualityConfig(
            min_image_width=100, min_image_height=100, max_image_width=5000, max_image_height=5000
        )
        return ImageValidator(cfg)

    @pytest.mark.asyncio
    async def test_valid_images(
        self, validator: ImageValidator, sample_images: Sequence[ImageInfo]
    ) -> None:
        result = await validator.validate(sample_images)
        assert result.passed is True
        assert result.total_images == 5

    @pytest.mark.asyncio
    async def test_empty_images(self, validator: ImageValidator) -> None:
        result = await validator.validate([])
        assert result.passed is True
        assert result.total_images == 0

    @pytest.mark.asyncio
    async def test_tiny_image_detected(
        self, validator: ImageValidator, sample_tiny_image: ImageInfo
    ) -> None:
        result = await validator.validate([sample_tiny_image])
        assert result.tiny_image_count == 1
        assert len(result.issues) >= 1
        issue = result.issues[0]
        assert issue.category.value == "image"
        assert "too small" in issue.message.lower()

    @pytest.mark.asyncio
    async def test_large_image_detected(
        self, validator: ImageValidator, sample_large_image: ImageInfo
    ) -> None:
        result = await validator.validate([sample_large_image])
        assert result.oversized_image_count == 1
        assert len(result.issues) >= 1
        issue = result.issues[0]
        assert "large" in issue.message.lower()

    @pytest.mark.asyncio
    async def test_wrong_format_detected(
        self, validator: ImageValidator, sample_wrong_format_image: ImageInfo
    ) -> None:
        result = await validator.validate([sample_wrong_format_image])
        assert result.wrong_format_count == 1
        issue = result.issues[0]
        assert "format" in issue.message.lower()

    @pytest.mark.asyncio
    async def test_wrong_color_space_detected(
        self, validator: ImageValidator, sample_wrong_color_image: ImageInfo
    ) -> None:
        result = await validator.validate([sample_wrong_color_image])
        assert result.wrong_color_space_count == 1
        issue = result.issues[0]
        assert "color space" in issue.message.lower()

    @pytest.mark.asyncio
    async def test_strict_thresholds(
        self, strict_validator: ImageValidator, sample_images: Sequence[ImageInfo]
    ) -> None:
        result = await strict_validator.validate(sample_images)
        assert result.tiny_image_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_issues_on_one_image(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="multi_issue",
            file_path="/data/bad.bmp",
            width=10,
            height=10,
            format="bmp",
            color_space="cmyk",
        )
        result = await validator.validate([img])
        assert result.tiny_image_count == 1
        assert result.wrong_format_count == 1
        assert result.wrong_color_space_count == 1

    @pytest.mark.asyncio
    async def test_jpeg_format_accepted(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="jpeg",
            file_path="/data/img.jpeg",
            width=640,
            height=480,
            format="jpeg",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.wrong_format_count == 0

    @pytest.mark.asyncio
    async def test_png_format_accepted(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="png",
            file_path="/data/img.png",
            width=640,
            height=480,
            format="png",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.wrong_format_count == 0

    @pytest.mark.asyncio
    async def test_webp_format_accepted(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="webp",
            file_path="/data/img.webp",
            width=640,
            height=480,
            format="webp",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.wrong_format_count == 0

    @pytest.mark.asyncio
    async def test_bgr_accepted(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="bgr",
            file_path="/data/img.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="bgr",
        )
        result = await validator.validate([img])
        assert result.wrong_color_space_count == 0

    @pytest.mark.asyncio
    async def test_none_dimensions_skipped(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="no_dims",
            file_path="/data/img.jpg",
            width=None,
            height=None,
            format="jpg",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.tiny_image_count == 0
        assert result.oversized_image_count == 0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_none_format_skipped(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="no_fmt",
            file_path="/data/img.jpg",
            width=640,
            height=480,
            format=None,
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.wrong_format_count == 0

    @pytest.mark.asyncio
    async def test_tiff_format_warning(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="tiff",
            file_path="/data/img.tiff",
            width=640,
            height=480,
            format="tiff",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.wrong_format_count == 1

    @pytest.mark.asyncio
    async def test_gif_format_warning(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="gif",
            file_path="/data/img.gif",
            width=640,
            height=480,
            format="gif",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.wrong_format_count == 1

    @pytest.mark.asyncio
    async def test_cmyk_color_warning(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="cmyk",
            file_path="/data/img.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="cmyk",
        )
        result = await validator.validate([img])
        assert result.wrong_color_space_count == 1

    @pytest.mark.asyncio
    async def test_yuv_color_warning(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="yuv",
            file_path="/data/img.jpg",
            width=640,
            height=480,
            format="jpg",
            color_space="yuv",
        )
        result = await validator.validate([img])
        assert result.wrong_color_space_count == 1

    @pytest.mark.asyncio
    async def test_exact_min_dimensions(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="min",
            file_path="/data/min.jpg",
            width=32,
            height=32,
            format="jpg",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.tiny_image_count == 0

    @pytest.mark.asyncio
    async def test_exact_max_dimensions(self, validator: ImageValidator) -> None:
        img = ImageInfo(
            id="max",
            file_path="/data/max.jpg",
            width=10000,
            height=10000,
            format="jpg",
            color_space="rgb",
        )
        result = await validator.validate([img])
        assert result.oversized_image_count == 0

    @pytest.mark.asyncio
    async def test_mixed_valid_and_invalid(self, validator: ImageValidator) -> None:
        imgs = [
            ImageInfo(
                id="good",
                file_path="/data/good.jpg",
                width=640,
                height=480,
                format="jpg",
                color_space="rgb",
            ),
            ImageInfo(
                id="tiny",
                file_path="/data/tiny.jpg",
                width=16,
                height=16,
                format="jpg",
                color_space="rgb",
            ),
            ImageInfo(
                id="bad_fmt",
                file_path="/data/bad.bmp",
                width=640,
                height=480,
                format="bmp",
                color_space="rgb",
            ),
        ]
        result = await validator.validate(imgs)
        assert result.tiny_image_count == 1
        assert result.wrong_format_count == 1

    @pytest.mark.asyncio
    async def test_large_validator_config(self) -> None:
        cfg = DatasetQualityConfig(max_image_width=100000, max_image_height=100000)
        v = ImageValidator(cfg)
        img = ImageInfo(
            id="huge",
            file_path="/data/huge.jpg",
            width=50000,
            height=50000,
            format="jpg",
            color_space="rgb",
        )
        result = await v.validate([img])
        assert result.oversized_image_count == 0

    @pytest.mark.asyncio
    async def test_inherits_interface(self) -> None:
        from backend.dataset_quality.interfaces import ImageValidatorInterface

        assert issubclass(ImageValidator, ImageValidatorInterface)
