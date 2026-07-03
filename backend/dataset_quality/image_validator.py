from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import ImageInfo
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import ImageValidatorInterface
from backend.dataset_quality.models import (
    ImageValidationResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class ImageValidator(ImageValidatorInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

    async def validate(
        self,
        images: Sequence[ImageInfo],
        image_dir: str | None = None,
    ) -> ImageValidationResult:
        issues: list[QualityIssue] = []
        corrupt = 0
        unreadable = 0
        wrong_format = 0
        wrong_color = 0
        tiny = 0
        oversized = 0
        blank = 0

        for img in images:
            w, h = img.width, img.height

            if w is not None and h is not None:
                if w < self._config.min_image_width or h < self._config.min_image_height:
                    tiny += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.IMAGE,
                            message=f"Image {img.id} is too small ({w}x{h})",
                            location=img.file_path,
                            suggestion="Consider removing or upscaling tiny images",
                            details={
                                "width": w,
                                "height": h,
                                "min_w": self._config.min_image_width,
                                "min_h": self._config.min_image_height,
                            },
                        )
                    )

                if w > self._config.max_image_width or h > self._config.max_image_height:
                    oversized += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.IMAGE,
                            message=f"Image {img.id} is very large ({w}x{h})",
                            location=img.file_path,
                            suggestion="Consider downscaling very large images",
                            details={
                                "width": w,
                                "height": h,
                                "max_w": self._config.max_image_width,
                                "max_h": self._config.max_image_height,
                            },
                        )
                    )

            if img.format is not None and img.format.lower() not in ("jpg", "jpeg", "png", "webp"):
                wrong_format += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.IMAGE,
                        message=f"Image {img.id} has non-standard format: {img.format}",
                        location=img.file_path,
                        suggestion="Convert to JPEG or PNG for training compatibility",
                        details={"format": img.format},
                    )
                )

            if img.color_space is not None and img.color_space.lower() not in ("rgb", "bgr"):
                wrong_color += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.IMAGE,
                        message=f"Image {img.id} has non-RGB color space: {img.color_space}",
                        location=img.file_path,
                        suggestion="Convert to RGB color space",
                        details={"color_space": img.color_space},
                    )
                )

        total = len(images)
        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return ImageValidationResult(
            total_images=total,
            issues=tuple(issues),
            corrupt_count=corrupt,
            unreadable_count=unreadable,
            wrong_format_count=wrong_format,
            wrong_color_space_count=wrong_color,
            tiny_image_count=tiny,
            oversized_image_count=oversized,
            blank_image_count=blank,
            passed=passed,
        )
