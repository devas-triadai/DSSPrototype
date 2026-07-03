from __future__ import annotations

import hashlib
import os
from collections import defaultdict

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import DuplicateDetectorInterface
from backend.dataset_quality.models import (
    DuplicateDetectionResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class DuplicateDetector(DuplicateDetectorInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

    async def detect(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
    ) -> DuplicateDetectionResult:
        issues: list[QualityIssue] = []

        dup_image_pairs: list[tuple[str, str]] = []
        near_dup_image_pairs: list[tuple[str, str]] = []
        dup_ann_pairs: list[tuple[str, str]] = []
        repeated_ids: list[str] = []

        seen_hashes: dict[str, str] = {}
        for img in dataset.images:
            file_path = img.file_path
            if image_dir and not os.path.isabs(file_path):
                file_path = os.path.join(image_dir, file_path)

            content_hash = None
            if os.path.isfile(file_path):
                with open(file_path, "rb") as f:
                    content_hash = hashlib.sha256(f.read()).hexdigest()

            if content_hash:
                if content_hash in seen_hashes:
                    other_id = seen_hashes[content_hash]
                    dup_image_pairs.append((other_id, img.id))
                    issues.append(
                        QualityIssue(
                            severity=Severity.WARNING,
                            category=QualityCategory.DUPLICATE,
                            message=f"Image {img.id} duplicates {other_id} (same content hash)",
                            location=img.id,
                            suggestion="Remove duplicate images to avoid data leakage",
                            details={"duplicate_of": other_id, "hash": content_hash[:16]},
                        )
                    )
                else:
                    seen_hashes[content_hash] = img.id

        images_list = list(dataset.images)
        if images_list:
            first_img = images_list[0]
            if first_img.width is not None and first_img.height is not None:
                for i in range(len(images_list)):
                    for j in range(i + 1, len(images_list)):
                        a, b = images_list[i], images_list[j]
                        if a.width == b.width and a.height == b.height and a.format == b.format:
                            if a.file_path != b.file_path:
                                near_dup_image_pairs.append((a.id, b.id))

        for pair in near_dup_image_pairs:
            issues.append(
                QualityIssue(
                    severity=Severity.INFO,
                    category=QualityCategory.DUPLICATE,
                    message=f"Images {pair[0]} and {pair[1]} may be near-duplicates (same dims/format)",  # noqa: E501
                    location=pair[0],
                    suggestion="Verify these images are distinct; remove if duplicated",
                    details={"image_a": pair[0], "image_b": pair[1]},
                )
            )

        ann_signatures: dict[tuple[object, ...], str] = {}
        for ann in dataset.annotations:
            sig = (ann.image_id, ann.x, ann.y, ann.width, ann.height, ann.canonical_label)
            if sig in ann_signatures:
                other_id = ann_signatures[sig]
                dup_ann_pairs.append((other_id, ann.id))
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.DUPLICATE,
                        message=f"Annotation {ann.id} duplicates {other_id} (same image, label, coords)",  # noqa: E501
                        location=ann.id,
                        suggestion="Remove duplicate annotations",
                        details={"duplicate_of": other_id},
                    )
                )
            else:
                ann_signatures[sig] = ann.id

        all_ids: list[str] = []
        for img in dataset.images:
            all_ids.append(img.id)
        for ann in dataset.annotations:
            all_ids.append(ann.id)
        id_counts: dict[str, int] = defaultdict(int)
        for id_val in all_ids:
            id_counts[id_val] += 1
        for id_val, count in id_counts.items():
            if count > 1:
                repeated_ids.append(id_val)
                issues.append(
                    QualityIssue(
                        severity=Severity.ERROR,
                        category=QualityCategory.DUPLICATE,
                        message=f"ID '{id_val}' appears {count} times",
                        location=id_val,
                        suggestion="Ensure all IDs are unique across images and annotations",
                        details={"count": count},
                    )
                )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return DuplicateDetectionResult(
            issues=tuple(issues),
            duplicate_image_pairs=tuple(dup_image_pairs),
            duplicate_annotation_pairs=tuple(dup_ann_pairs),
            near_duplicate_image_pairs=tuple(near_dup_image_pairs),
            repeated_ids=tuple(repeated_ids),
            total_duplicate_images=len(dup_image_pairs),
            total_duplicate_annotations=len(dup_ann_pairs),
            passed=passed,
        )
