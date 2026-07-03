from __future__ import annotations

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.interfaces import AnnotationValidatorInterface
from backend.dataset_quality.models import (
    AnnotationValidationResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class AnnotationValidator(AnnotationValidatorInterface):
    async def validate(
        self,
        dataset: CanonicalDataset,
    ) -> AnnotationValidationResult:
        issues: list[QualityIssue] = []
        neg_coords = 0
        zero_area = 0
        out_of_bounds = 0
        invalid_poly = 0
        broken_seg = 0
        broken_obb = 0

        image_map: dict[str, tuple[int | None, int | None]] = {}
        for img in dataset.images:
            image_map[img.id] = (img.width, img.height)

        annotated_image_ids: set[str] = set()
        for ann in dataset.annotations:
            annotated_image_ids.add(ann.image_id)

            if ann.x < 0 or ann.y < 0:
                neg_coords += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.ERROR,
                        category=QualityCategory.ANNOTATION,
                        message=f"Annotation {ann.id} has negative coordinates ({ann.x}, {ann.y})",
                        location=ann.id,
                        suggestion="Clip or remove annotations with negative coordinates",
                        details={"x": ann.x, "y": ann.y},
                    )
                )

            if ann.width <= 0 or ann.height <= 0:
                zero_area += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.ERROR,
                        category=QualityCategory.ANNOTATION,
                        message=f"Annotation {ann.id} has zero/negative area ({ann.width}x{ann.height})",  # noqa: E501
                        location=ann.id,
                        suggestion="Remove zero-area annotations",
                        details={"width": ann.width, "height": ann.height},
                    )
                )

            img_w, img_h = image_map.get(ann.image_id, (None, None))
            if img_w is not None and img_h is not None:
                if ann.x + ann.width > img_w or ann.y + ann.height > img_h:
                    out_of_bounds += 1
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.ANNOTATION,
                            message=f"Annotation {ann.id} extends beyond image bounds",
                            location=ann.id,
                            suggestion="Clip annotation coordinates to image boundaries",
                            details={
                                "x": ann.x,
                                "y": ann.y,
                                "w": ann.width,
                                "h": ann.height,
                                "img_w": img_w,
                                "img_h": img_h,
                            },
                        )
                    )

            if ann.geometry_type.value == "polygon" and ann.width < 1 and ann.height < 1:
                invalid_poly += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.ANNOTATION,
                        message=f"Annotation {ann.id} has invalid polygon geometry",
                        location=ann.id,
                        suggestion="Fix or remove polygon with invalid shape",
                        details={"geometry_type": ann.geometry_type.value},
                    )
                )

            if ann.geometry_type.value == "obb" and (ann.width <= 0 or ann.height <= 0):
                broken_obb += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.ANNOTATION,
                        message=f"Annotation {ann.id} has invalid OBB geometry",
                        location=ann.id,
                        suggestion="Fix or remove OBB with invalid dimensions",
                        details={"width": ann.width, "height": ann.height},
                    )
                )

        missing_ann_images: list[str] = []
        for img in dataset.images:
            if img.id not in annotated_image_ids:
                missing_ann_images.append(img.id)

        all_issues: list[QualityIssue] = list(issues)
        for img_id in missing_ann_images:
            all_issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.ANNOTATION,
                    message=f"Image {img_id} has no annotations",
                    location=img_id,
                    suggestion="Consider whether this image should be in the dataset",
                )
            )

        passed = len([x for x in all_issues if x.severity == Severity.ERROR]) == 0

        return AnnotationValidationResult(
            total_annotations=dataset.annotation_count,
            total_images=dataset.image_count,
            issues=tuple(all_issues),
            missing_annotation_count=len(missing_ann_images),
            negative_coordinate_count=neg_coords,
            zero_area_count=zero_area,
            out_of_bounds_count=out_of_bounds,
            invalid_polygon_count=invalid_poly,
            broken_segmentation_count=broken_seg,
            broken_obb_count=broken_obb,
            images_without_annotations=tuple(missing_ann_images),
            passed=passed,
        )
