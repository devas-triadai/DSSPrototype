from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import ClassValidatorInterface
from backend.dataset_quality.models import (
    ClassValidationResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class ClassValidator(ClassValidatorInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

    async def validate(
        self,
        dataset: CanonicalDataset,
        ontology_classes: Sequence[str] | None = None,
    ) -> ClassValidationResult:
        issues: list[QualityIssue] = []
        class_counts: dict[str, int] = {}
        for ann in dataset.annotations:
            class_counts[ann.canonical_label] = class_counts.get(ann.canonical_label, 0) + 1

        ontology_set: set[str] = set(ontology_classes) if ontology_classes else set()
        unknown = 0
        ontology_mismatch = 0

        for cls_name in class_counts:
            if ontology_set and cls_name not in ontology_set:
                unknown += 1
                issues.append(
                    QualityIssue(
                        severity=Severity.ERROR,
                        category=QualityCategory.CLASS,
                        message=f"Unknown class '{cls_name}' not found in ontology",
                        location=cls_name,
                        suggestion="Map this class to a valid ontology value or remove it",
                        details={"count": class_counts[cls_name]},
                    )
                )

        found_set = set(class_counts.keys())
        if ontology_set:
            unused = sorted(ontology_set - found_set)
            for cls_name in unused:
                issues.append(
                    QualityIssue(
                        severity=Severity.INFO,
                        category=QualityCategory.CLASS,
                        message=f"Ontology class '{cls_name}' has no samples in the dataset",
                        location=cls_name,
                        suggestion="Add samples for this class or remove it from the ontology",
                    )
                )
        else:
            unused = []

        if ontology_set and found_set:
            ontology_mismatch = len(found_set - ontology_set) + len(ontology_set - found_set)

        rare_threshold = max(
            self._config.rare_class_threshold * dataset.annotation_count,
            self._config.min_class_samples,
        )
        rare_classes: list[str] = []
        for cls_name, count in class_counts.items():
            if count < rare_threshold:
                rare_classes.append(cls_name)
                issues.append(
                    QualityIssue(
                        severity=Severity.WARNING,
                        category=QualityCategory.CLASS,
                        message=f"Class '{cls_name}' has {count} samples (threshold: {int(rare_threshold)})",  # noqa: E501
                        location=cls_name,
                        suggestion="Consider collecting more samples or applying augmentation",
                        details={"count": count, "threshold": int(rare_threshold)},
                    )
                )

        imbalance_ratio = 1.0
        if class_counts and len(class_counts) > 1:
            counts = list(class_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            if min_count > 0:
                imbalance_ratio = max_count / min_count

        if imbalance_ratio > 10:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.CLASS,
                    message=f"Class imbalance ratio is {imbalance_ratio:.1f}:1",
                    location="dataset",
                    suggestion="Use class-weighted loss or augment minority classes",
                    details={"imbalance_ratio": imbalance_ratio},
                )
            )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return ClassValidationResult(
            total_classes=len(class_counts),
            issues=tuple(issues),
            unknown_class_count=unknown,
            ontology_mismatch_count=ontology_mismatch,
            unused_class_count=len(unused),
            rare_class_count=len(rare_classes),
            class_distribution=class_counts,
            imbalance_ratio=imbalance_ratio,
            passed=passed,
        )
