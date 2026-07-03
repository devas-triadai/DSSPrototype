from __future__ import annotations

import os

from backend.dataset_conversion.config import DatasetConversionConfig, dataset_conversion_config
from backend.dataset_conversion.interfaces import ImageConverterInterface
from backend.dataset_conversion.models import ImageInfo


class ImageConverter(ImageConverterInterface):
    def __init__(self, config: DatasetConversionConfig | None = None) -> None:
        self._config = config or dataset_conversion_config

    async def validate_image(self, image: ImageInfo) -> list[str]:
        errors: list[str] = []

        if not image.file_path:
            errors.append(f"Image '{image.id}' has no file path")
            return errors

        if not os.path.isfile(image.file_path):
            errors.append(f"Image file not found: {image.file_path}")
            return errors

        if self._config.max_image_width > 0 and image.width is not None:
            if image.width > self._config.max_image_width:
                errors.append(
                    f"Image '{image.id}' width {image.width} exceeds "
                    f"maximum {self._config.max_image_width}"
                )

        if self._config.max_image_height > 0 and image.height is not None:
            if image.height > self._config.max_image_height:
                errors.append(
                    f"Image '{image.id}' height {image.height} exceeds "
                    f"maximum {self._config.max_image_height}"
                )

        valid_formats = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
        ext = os.path.splitext(image.file_path)[1].lower().lstrip(".")
        if ext and ext not in valid_formats:
            errors.append(f"Image '{image.id}' format '{ext}' is not supported")

        return errors

    async def standardize_metadata(self, image: ImageInfo) -> ImageInfo:
        ext = (
            os.path.splitext(image.file_path)[1].lower().lstrip(".")
            if image.file_path
            else image.format
        )
        return ImageInfo(
            id=image.id,
            file_path=image.file_path,
            width=image.width,
            height=image.height,
            format=ext or image.format or self._config.target_image_format,
            color_space=image.color_space or self._config.target_color_space,
            file_size_bytes=self._get_file_size(image.file_path),
            metadata=dict(image.metadata),
        )

    def _get_file_size(self, path: str) -> int | None:
        try:
            return os.path.getsize(path) if os.path.isfile(path) else None
        except OSError:
            return None
