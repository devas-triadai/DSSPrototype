from __future__ import annotations

import statistics

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import OutlierDetectorInterface
from backend.dataset_quality.models import (
    OutlierDetectionResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class OutlierDetector(OutlierDetectorInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

    async def detect(
        self,
        dataset: CanonicalDataset,
    ) -> OutlierDetectionResult:
        issues: list[QualityIssue] = []
        extreme_ar: list[dict[str, object]] = []
        tiny: list[dict[str, object]] = []
        huge: list[dict[str, object]] = []
        suspicious: list[dict[str, object]] = []
        abnormal_res: list[dict[str, object]] = []

        image_dims: list[tuple[int, int]] = []
        for img in dataset.images:
            if img.width is not None and img.height is not None:
                image_dims.append((img.width, img.height))

        if image_dims:
            widths = [w for w, h in image_dims]
            heights = [h for w, h in image_dims]
            mean_w = statistics.mean(widths) if widths else 0
            mean_h = statistics.mean(heights) if heights else 0
            std_w = statistics.stdev(widths) if len(widths) > 1 else 0
            std_h = statistics.stdev(heights) if len(heights) > 1 else 0

            threshold = self._config.outlier_std_dev_threshold
            for img in dataset.images:
                if img.width is None or img.height is None:
                    continue
                if std_w > 0 and abs(img.width - mean_w) / std_w > threshold:
                    abnormal_res.append(
                        {
                            "image_id": img.id,
                            "width": img.width,
                            "height": img.height,
                            "z_score_w": (img.width - mean_w) / std_w,
                        }
                    )
                if std_h > 0 and abs(img.height - mean_h) / std_h > threshold:
                    if not any(d.get("image_id") == img.id for d in abnormal_res):
                        abnormal_res.append(
                            {
                                "image_id": img.id,
                                "width": img.width,
                                "height": img.height,
                                "z_score_h": (img.height - mean_h) / std_h,
                            }
                        )

        for entry in abnormal_res:
            issues.append(
                QualityIssue(
                    severity=Severity.INFO,
                    category=QualityCategory.OUTLIER,
                    message=f"Image {entry['image_id']} abnormal resolution ({entry['width']}x{entry['height']})",  # noqa: E501
                    location=str(entry["image_id"]),
                    suggestion="Verify the image resolution is appropriate for the dataset",
                    details=entry,
                )
            )

        annotation_areas: list[tuple[str, float]] = []
        for ann in dataset.annotations:
            ann_area = ann.width * ann.height
            annotation_areas.append((ann.id, ann_area))

            aspect_ratio = ann.height / ann.width if ann.width > 0 else float("inf")
            if (
                aspect_ratio > self._config.max_aspect_ratio
                or (1 / aspect_ratio if aspect_ratio > 0 else float("inf"))
                > self._config.max_aspect_ratio
            ):
                extreme_ar.append(
                    {
                        "annotation_id": ann.id,
                        "width": ann.width,
                        "height": ann.height,
                        "aspect_ratio": aspect_ratio,
                    }
                )

            if ann_area < self._config.min_object_area:
                tiny.append(
                    {
                        "annotation_id": ann.id,
                        "label": ann.canonical_label,
                        "width": ann.width,
                        "height": ann.height,
                        "area": ann_area,
                    }
                )

        for entry in extreme_ar:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.OUTLIER,
                    message=f"Annotation {entry['annotation_id']} extreme aspect ratio ({entry['aspect_ratio']:.2f})",  # noqa: E501
                    location=str(entry["annotation_id"]),
                    suggestion="Verify the annotation geometry is correct",
                    details=entry,
                )
            )

        for entry in tiny:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.OUTLIER,
                    message=f"Tiny object {entry['annotation_id']} ({entry['label']}) area {entry['area']:.1f}",  # noqa: E501
                    location=str(entry["annotation_id"]),
                    suggestion="Consider whether tiny objects are relevant for training",
                    details=entry,
                )
            )

        if annotation_areas:
            areas_values = [a for _, a in annotation_areas]
            mean_area = statistics.mean(areas_values) if areas_values else 0
            std_area = statistics.stdev(areas_values) if len(areas_values) > 1 else 0
            threshold_area = mean_area + self._config.outlier_std_dev_threshold * std_area

            for ann_id, area in annotation_areas:
                if std_area > 0 and area > threshold_area:
                    huge.append(
                        {
                            "annotation_id": ann_id,
                            "area": area,
                            "threshold": threshold_area,
                        }
                    )

        for entry in huge:
            issues.append(
                QualityIssue(
                    severity=Severity.INFO,
                    category=QualityCategory.OUTLIER,
                    message=f"Large object {entry['annotation_id']} (area {entry['area']:.1f})",
                    location=str(entry["annotation_id"]),
                    suggestion="Verify the annotation covers a single object",
                    details=entry,
                )
            )

        total_outliers = (
            len(extreme_ar) + len(tiny) + len(huge) + len(suspicious) + len(abnormal_res)
        )
        passed = True

        return OutlierDetectionResult(
            issues=tuple(issues),
            extreme_aspect_ratios=tuple(extreme_ar),
            tiny_objects=tuple(tiny),
            huge_objects=tuple(huge),
            suspicious_annotations=tuple(suspicious),
            abnormal_resolutions=tuple(abnormal_res),
            total_outliers=total_outliers,
            passed=passed,
        )
