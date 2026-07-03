from __future__ import annotations

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.interfaces import ConsistencyCheckerInterface
from backend.dataset_quality.models import (
    ConsistencyCheckResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class ConsistencyChecker(ConsistencyCheckerInterface):
    async def check(
        self,
        dataset: CanonicalDataset,
    ) -> ConsistencyCheckResult:
        issues: list[QualityIssue] = []

        versions = {img.metadata.get("pipeline_version", "") for img in dataset.images}
        metadata_consistent = len(versions) <= 1
        if not metadata_consistent:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.CONSISTENCY,
                    message=f"Inconsistent pipeline versions found: {versions}",
                    location="dataset",
                    suggestion="Re-process all images with the same pipeline version",
                    details={"versions": sorted(versions)},
                )
            )

        split_consistent = True
        metadata = dataset.metadata
        if "split" in metadata:
            split_value = metadata["split"]
            annotations_with_split = [
                ann
                for ann in dataset.annotations
                if ann.metadata.get("split", split_value) != split_value
            ]
            if annotations_with_split:
                split_consistent = False
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.CONSISTENCY,
                        message="Split labels are inconsistent across annotations",
                        location="dataset",
                        suggestion="Verify the dataset split assignment",
                    )
                )

        label_set = {ann.canonical_label for ann in dataset.annotations}
        ontology_consistent = len(label_set) > 0
        if not ontology_consistent:
            issues.append(
                QualityIssue(
                    severity=Severity.ERROR,
                    category=QualityCategory.CONSISTENCY,
                    message="No valid ontology labels found in annotations",
                    location="dataset",
                    suggestion="Ensure annotations have valid canonical labels",
                )
            )

        ann_label_map: dict[str, set[str]] = {}
        for ann in dataset.annotations:
            if ann.image_id not in ann_label_map:
                ann_label_map[ann.image_id] = set()
            ann_label_map[ann.image_id].add(ann.canonical_label)

        annotation_consistent = True
        if len(ann_label_map) > 1:
            label_sets = list(ann_label_map.values())
            ref_set = label_sets[0]
            for ls in label_sets[1:]:
                common = ref_set & ls
                if len(common) < min(len(ref_set), len(ls)) * 0.5:
                    annotation_consistent = False
                    break

        if not annotation_consistent:
            issues.append(
                QualityIssue(
                    severity=Severity.INFO,
                    category=QualityCategory.CONSISTENCY,
                    message="Annotation label distributions vary significantly across images",
                    location="dataset",
                    suggestion="Verify the labeling scheme is applied uniformly",
                )
            )

        version_consistent = True
        if not dataset.ontology_version or not dataset.pipeline_version:
            version_consistent = False
            issues.append(
                QualityIssue(
                    severity=Severity.ERROR,
                    category=QualityCategory.CONSISTENCY,
                    message="Dataset is missing ontology or pipeline version",
                    location="dataset",
                    suggestion="Ensure both ontology_version and pipeline_version are set",
                )
            )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return ConsistencyCheckResult(
            metadata_consistent=metadata_consistent,
            split_consistent=split_consistent,
            ontology_consistent=ontology_consistent,
            annotation_consistent=annotation_consistent,
            version_consistent=version_consistent,
            issues=tuple(issues),
            passed=passed,
        )
