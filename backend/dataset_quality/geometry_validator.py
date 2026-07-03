from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import CanonicalAnnotation, GeometryType
from backend.dataset_quality.interfaces import GeometryValidatorInterface
from backend.dataset_quality.models import (
    GeometryValidationResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class GeometryValidator(GeometryValidatorInterface):
    async def validate(
        self,
        annotations: Sequence[CanonicalAnnotation],
    ) -> GeometryValidationResult:
        issues: list[QualityIssue] = []
        invalid_bbox = 0
        invalid_polygon = 0
        invalid_seg = 0
        invalid_rotated = 0
        invalid_normalized = 0
        invalid_pixel = 0

        for ann in annotations:
            gt = ann.geometry_type

            if gt == GeometryType.BBOX:
                if ann.width <= 0 or ann.height <= 0:
                    invalid_bbox += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.ERROR,
                            category=QualityCategory.GEOMETRY,
                            message=f"Annotation {ann.id} has invalid bbox ({ann.width}x{ann.height})",  # noqa: E501
                            location=ann.id,
                            suggestion="Fix or remove bbox with zero/negative dimensions",
                            details={"width": ann.width, "height": ann.height},
                        )
                    )

            elif gt == GeometryType.POLYGON:
                if ann.width <= 0 or ann.height <= 0:
                    invalid_polygon += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.GEOMETRY,
                            message=f"Annotation {ann.id} has invalid polygon ({ann.width}x{ann.height})",  # noqa: E501
                            location=ann.id,
                            suggestion="Check polygon coordinates for correctness",
                            details={"width": ann.width, "height": ann.height},
                        )
                    )

            elif gt == GeometryType.SEGMENTATION:
                if ann.width <= 0 or ann.height <= 0:
                    invalid_seg += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.GEOMETRY,
                            message=f"Annotation {ann.id} has invalid segmentation extent",
                            location=ann.id,
                            suggestion="Verify segmentation mask dimensions",
                            details={"width": ann.width, "height": ann.height},
                        )
                    )

            elif gt == GeometryType.OBB:
                if ann.width <= 0 or ann.height <= 0:
                    invalid_rotated += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.GEOMETRY,
                            message=f"Annotation {ann.id} has invalid rotated box dimensions",
                            location=ann.id,
                            suggestion="Fix rotated box width/height",
                            details={"width": ann.width, "height": ann.height},
                        )
                    )

            if ann.x < 0 or ann.y < 0:
                invalid_pixel += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.GEOMETRY,
                        message=f"Annotation {ann.id} has negative pixel coordinates",
                        location=ann.id,
                        suggestion="Clip coordinates to non-negative range",
                        details={"x": ann.x, "y": ann.y},
                    )
                )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return GeometryValidationResult(
            total_geometries=len(annotations),
            issues=tuple(issues),
            invalid_bbox_count=invalid_bbox,
            invalid_polygon_count=invalid_polygon,
            invalid_segmentation_count=invalid_seg,
            invalid_rotated_box_count=invalid_rotated,
            invalid_normalized_coord_count=invalid_normalized,
            invalid_pixel_coord_count=invalid_pixel,
            passed=passed,
        )
