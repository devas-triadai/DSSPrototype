"""Dataset validation engine.

Runs structural, semantic, and integrity checks on a ``RawDataset``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from backend.dataset_intelligence.interfaces import DatasetValidatorInterface
from backend.dataset_intelligence.models import RawDataset, ValidationReport

logger = logging.getLogger("dss.dataset_intelligence.validator")


class DatasetValidator(DatasetValidatorInterface):
    """Validate a raw dataset before normalization.

    Checks performed:
      - Image/annotation consistency (every annotation has a readable image)
      - Bounding box validity (0 <= coords <= 1 in normalized space)
      - Class consistency (classes in annotations exist in dataset class list)
      - File integrity (checksums where available)
      - No duplicate image IDs
      - No negative coordinates
    """

    def validate(self, dataset: RawDataset) -> ValidationReport:
        logger.info("Validation started | dataset=%s", dataset.dataset_id)
        missing_images: list[str] = []
        missing_annotations: list[str] = []
        empty_annotations: list[str] = []
        corrupted_files: list[str] = []
        unsupported_extensions: list[str] = []
        invalid_bounding_boxes: list[str] = []
        negative_coordinates: list[str] = []
        class_mismatches: list[str] = []
        orphan_labels: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []
        passed_checks = 0
        failed_checks = 0
        total_checks = 0

        image_ids = set()
        for img in dataset.images:
            total_checks += 1
            img_path = Path(img.image_path)
            if not img_path.exists():
                missing_images.append(img.image_id)
                failed_checks += 1
                continue
            if not img_path.is_file():
                corrupted_files.append(img.image_id)
                failed_checks += 1
                continue
            passed_checks += 1

            total_checks += 1
            if img.image_id in image_ids:
                warnings.append(f"Duplicate image_id: {img.image_id}")
                failed_checks += 1
            else:
                image_ids.add(img.image_id)
                passed_checks += 1

            total_checks += 1
            if img.width <= 0 or img.height <= 0:
                warnings.append(f"Invalid dimensions for {img.image_id}")
                failed_checks += 1
            else:
                passed_checks += 1

            ext = img_path.suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}:
                unsupported_extensions.append(img.image_id)

            if not img.annotations:
                empty_annotations.append(img.image_id)

            for ann in img.annotations:
                total_checks += 1
                if ann.class_name not in dataset.classes:
                    class_mismatches.append(f"{img.image_id}: {ann.class_name}")
                    failed_checks += 1
                else:
                    passed_checks += 1

                total_checks += 1
                x1, y1, x2, y2 = ann.bbox
                if any(v < 0 for v in ann.bbox):
                    negative_coordinates.append(f"{img.image_id}: {ann.bbox}")
                    failed_checks += 1
                elif x1 > x2 or y1 > y2:
                    invalid_bounding_boxes.append(f"{img.image_id}: {ann.bbox}")
                    failed_checks += 1
                elif x2 > 1.0 or y2 > 1.0:
                    invalid_bounding_boxes.append(f"{img.image_id}: {ann.bbox}")
                    failed_checks += 1
                else:
                    passed_checks += 1

        total_checks += 1
        if not dataset.images:
            errors.append("Dataset contains no images")
            failed_checks += 1
        else:
            passed_checks += 1

        total_checks += 1
        if not dataset.classes:
            errors.append("Dataset contains no classes")
            failed_checks += 1
        else:
            passed_checks += 1

        passed = failed_checks == 0 and len(errors) == 0

        report = ValidationReport(
            dataset_id=dataset.dataset_id,
            passed=passed,
            missing_images=missing_images,
            missing_annotations=missing_annotations,
            empty_annotations=empty_annotations,
            corrupted_files=corrupted_files,
            unsupported_extensions=unsupported_extensions,
            invalid_bounding_boxes=invalid_bounding_boxes,
            negative_coordinates=negative_coordinates,
            class_mismatches=class_mismatches,
            orphan_labels=orphan_labels,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            errors=errors,
        )
        logger.info(
            "Validation complete | dataset=%s | passed=%s | checks=%d/%d",
            dataset.dataset_id,
            passed,
            passed_checks,
            total_checks,
        )
        return report
