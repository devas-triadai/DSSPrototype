from __future__ import annotations

import hashlib
import os

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.interfaces import IntegrityCheckerInterface
from backend.dataset_quality.models import (
    IntegrityCheckResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class IntegrityChecker(IntegrityCheckerInterface):
    async def check(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
    ) -> IntegrityCheckResult:
        issues: list[QualityIssue] = []
        missing_files: list[str] = []
        broken_refs: list[str] = []
        checksums_valid = True
        manifest_valid = True
        all_present = True
        no_broken = True
        version_valid = True

        valid_image_ids = {img.id for img in dataset.images}

        image_file_map: dict[str, str] = {}
        for img in dataset.images:
            file_path = img.file_path
            if image_dir and not os.path.isabs(file_path):
                full_path = os.path.join(image_dir, file_path)
            else:
                full_path = file_path

            if not os.path.isfile(full_path):
                missing_files.append(file_path)
                all_present = False
            else:
                image_file_map[img.id] = full_path

        for file_path in missing_files:
            issues.append(
                QualityIssue(
                    severity=Severity.ERROR,
                    category=QualityCategory.INTEGRITY,
                    message=f"Image file not found: {file_path}",
                    location=file_path,
                    suggestion="Ensure all image files exist at the expected paths",
                    details={"file_path": file_path},
                )
            )

        ann_image_ids = {ann.image_id for ann in dataset.annotations}
        for ann_id in ann_image_ids:
            if ann_id not in valid_image_ids:
                broken_refs.append(ann_id)

        for ref in broken_refs:
            issues.append(
                QualityIssue(
                    severity=Severity.ERROR,
                    category=QualityCategory.INTEGRITY,
                    message=f"Broken reference: annotation points to non-existent image '{ref}'",
                    location=ref,
                    suggestion="Remove annotations that reference missing images",
                    details={"image_id": ref},
                )
            )

        no_broken = len(broken_refs) == 0

        if image_file_map:
            for img_id, path in image_file_map.items():
                try:
                    with open(path, "rb") as f:
                        actual_hash = hashlib.sha256(f.read()).hexdigest()
                    expected_hash = None
                    for img in dataset.images:
                        if img.id == img_id:
                            expected_hash = img.metadata.get("sha256")
                            break
                    if expected_hash and actual_hash != expected_hash:
                        checksums_valid = False
                        issues.append(
                            QualityIssue(
                                severity=Severity.ERROR,
                                category=QualityCategory.INTEGRITY,
                                message=f"Checksum mismatch for image {img_id}",
                                location=img_id,
                                suggestion="Re-export the dataset to regenerate checksums",
                                details={"expected": expected_hash, "actual": actual_hash[:16]},
                            )
                        )
                except (OSError, IOError):
                    pass

        if not dataset.ontology_version or not dataset.pipeline_version:
            version_valid = False
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.INTEGRITY,
                    message="Dataset version information is incomplete",
                    location="dataset",
                    suggestion="Ensure ontology_version and pipeline_version are set",
                    details={
                        "ontology_version": dataset.ontology_version,
                        "pipeline_version": dataset.pipeline_version,
                    },
                )
            )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return IntegrityCheckResult(
            checksums_valid=checksums_valid,
            manifest_valid=manifest_valid,
            all_files_present=all_present,
            no_broken_references=no_broken,
            version_valid=version_valid,
            missing_files=tuple(missing_files),
            broken_references=tuple(broken_refs),
            issues=tuple(issues),
            passed=passed,
        )
