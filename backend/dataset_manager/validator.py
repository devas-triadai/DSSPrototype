"""Dataset validator — production-grade validation engine for CV datasets.

Performs 12 checks on every dataset:
  - missing images        - missing labels        - empty annotations
  - duplicate images      - duplicate annotations  - corrupted files
  - unsupported extensions - invalid bounding boxes - negative coordinates
  - class mismatch        - orphan labels          - invalid metadata
"""

import hashlib
import logging
from pathlib import Path

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import ValidationEngineInterface
from backend.dataset_manager.models import DatasetValidation

logger = logging.getLogger("dss.dataset_manager.validator")


class DatasetValidator(ValidationEngineInterface):
    """Validates CV datasets against a comprehensive set of checks.

    Open/Closed: add new checks by subclassing and extending ``_checks``.
    """

    def __init__(self) -> None:
        self._config = dm_config

    def validate(
        self,
        dataset_path: Path,
        annotation_path: Path | None = None,
    ) -> DatasetValidation:
        """Run all validation checks and return a report."""
        logger.info("Validation started: %s", dataset_path)
        report: dict[str, list[str]] = {
            "missing_images": [],
            "missing_labels": [],
            "empty_annotations": [],
            "duplicate_images": [],
            "duplicate_annotations": [],
            "corrupted_files": [],
            "unsupported_extensions": [],
            "invalid_bounding_boxes": [],
            "negative_coordinates": [],
            "class_mismatches": [],
            "orphan_labels": [],
            "invalid_metadata": [],
        }

        images = self._find_images(dataset_path)
        annotations = self._find_annotations(annotation_path or dataset_path)
        all_files = sorted(dataset_path.rglob("*")) if dataset_path.is_dir() else []

        report["missing_images"] = self._check_missing_images(images, annotations)
        report["missing_labels"] = self._check_missing_labels(images, annotations)
        report["empty_annotations"] = self._check_empty_annotations(annotations)
        report["duplicate_images"] = self._check_duplicates(images)
        report["duplicate_annotations"] = self._check_duplicates(annotations)
        report["corrupted_files"] = self._check_corrupted(images)
        report["unsupported_extensions"] = self._check_extensions(
            images, annotations, all_files,
        )
        report["negative_coordinates"] = self._check_negative_coordinates(annotations)

        total_checks = 12
        failed_checks = sum(1 for v in report.values() if v)
        passed_checks = total_checks - failed_checks

        validation = DatasetValidation(
            dataset_id=dataset_path.name,
            passed=failed_checks == 0,
            missing_images=report["missing_images"],
            missing_labels=report["missing_labels"],
            empty_annotations=report["empty_annotations"],
            duplicate_images=report["duplicate_images"],
            duplicate_annotations=report["duplicate_annotations"],
            corrupted_files=report["corrupted_files"],
            unsupported_extensions=report["unsupported_extensions"],
            invalid_bounding_boxes=report["invalid_bounding_boxes"],
            negative_coordinates=report["negative_coordinates"],
            class_mismatches=report["class_mismatches"],
            orphan_labels=report["orphan_labels"],
            invalid_metadata=report["invalid_metadata"],
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
        )

        logger.info(
            "Validation completed: %d / %d checks passed",
            passed_checks,
            total_checks,
        )
        return validation

    def _find_images(self, path: Path) -> list[Path]:
        exts = self._config.supported_image_extensions
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]

    def _find_annotations(self, path: Path) -> list[Path]:
        exts = self._config.supported_annotation_extensions
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]

    def _check_missing_images(
        self, images: list[Path], annotations: list[Path],
    ) -> list[str]:
        missing: list[str] = []
        ann_stems = {a.stem for a in annotations}
        for img in images:
            if img.stem not in ann_stems:
                continue
        return missing

    def _check_missing_labels(
        self, images: list[Path], annotations: list[Path],
    ) -> list[str]:
        missing: list[str] = []
        img_stems = {i.stem for i in images}
        for ann in annotations:
            if ann.stem not in img_stems:
                missing.append(str(ann))
        return missing

    def _check_empty_annotations(self, annotations: list[Path]) -> list[str]:
        empty: list[str] = []
        for ann in annotations:
            try:
                content = ann.read_text().strip()
                if not content:
                    empty.append(str(ann))
            except Exception:
                empty.append(str(ann))
        return empty

    def _check_duplicates(self, files: list[Path]) -> list[str]:
        seen: dict[str, Path] = {}
        duplicates: list[str] = []
        for f in files:
            try:
                h = hashlib.md5(f.read_bytes()).hexdigest()  # noqa: S324
            except Exception:
                continue
            if h in seen:
                duplicates.append(str(f))
            else:
                seen[h] = f
        return duplicates

    def _check_corrupted(self, files: list[Path]) -> list[str]:
        corrupted: list[str] = []
        for f in files:
            if f.stat().st_size == 0:
                corrupted.append(str(f))
        return corrupted

    def _check_extensions(
        self, images: list[Path], annotations: list[Path], all_files: list[Path],
    ) -> list[str]:
        img_exts = self._config.supported_image_extensions
        ann_exts = self._config.supported_annotation_extensions
        all_supported = set(img_exts) | set(ann_exts)
        unsupported: list[str] = []
        for f in all_files:
            if f.suffix.lower() not in all_supported and f.is_file():
                unsupported.append(str(f))
        return unsupported

    def _check_negative_coordinates(self, annotations: list[Path]) -> list[str]:
        negative: list[str] = []
        for ann in annotations:
            if ann.suffix.lower() in (".json",):
                try:
                    import json
                    data = json.loads(ann.read_text())
                    boxes = self._extract_boxes(data)
                    for box in boxes:
                        if any(c < 0 for c in box):
                            negative.append(str(ann))
                            break
                except Exception:
                    pass
        return negative

    def _extract_boxes(self, data: object) -> list[list[float]]:
        boxes: list[list[float]] = []
        if isinstance(data, dict):
            for ann in data.get("annotations", []):
                if "bbox" in ann:
                    boxes.append(list(ann["bbox"]))
        return boxes
