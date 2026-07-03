from __future__ import annotations

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.config import DatasetQualityConfig, dataset_quality_config
from backend.dataset_quality.interfaces import ImbalanceAnalyzerInterface
from backend.dataset_quality.models import (
    ImbalanceAnalysisResult,
    QualityCategory,
    QualityIssue,
    Severity,
)


class ImbalanceAnalyzer(ImbalanceAnalyzerInterface):
    def __init__(self, config: DatasetQualityConfig | None = None):
        self._config = config or dataset_quality_config

    async def analyze(
        self,
        dataset: CanonicalDataset,
    ) -> ImbalanceAnalysisResult:
        issues: list[QualityIssue] = []
        class_counts: dict[str, int] = {}
        for ann in dataset.annotations:
            class_counts[ann.canonical_label] = class_counts.get(ann.canonical_label, 0) + 1

        total = sum(class_counts.values())
        num_classes = len(class_counts)

        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_dist: list[dict[str, object]] = [
            {"class": cls, "count": cnt, "proportion": round(cnt / total, 4) if total > 0 else 0}
            for cls, cnt in sorted_classes
        ]

        imbalance_ratio = 1.0
        majority_classes: list[str] = []
        minority_classes: list[str] = []

        if num_classes > 1 and total > 0:
            counts = list(class_counts.values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

            for cls, cnt in class_counts.items():
                if cnt >= max_count * 0.5:
                    majority_classes.append(cls)
                if (
                    cnt < self._config.min_class_samples
                    or cnt <= total * self._config.rare_class_threshold
                ):
                    minority_classes.append(cls)

        long_tail_ratio = 0.0
        if total > 0 and sorted_classes:
            top_k = max(1, num_classes // 3)
            top_count = sum(c for _, c in sorted_classes[:top_k])
            long_tail_ratio = top_count / total

        recommended_augmentations: dict[str, object] = {}
        if minority_classes:
            for cls in minority_classes:
                count = class_counts[cls]
                target = max(count * 3, self._config.min_class_samples * 2)
                recommended_augmentations[cls] = {
                    "current_count": count,
                    "target_count": target,
                    "techniques": ["random_flip", "color_jitter", "scale_variation"],
                }

        if imbalance_ratio > 10:
            issues.append(
                QualityIssue(
                    severity=Severity.WARNING,
                    category=QualityCategory.IMBALANCE,
                    message=f"High class imbalance ratio: {imbalance_ratio:.1f}:1",
                    location="dataset",
                    suggestion="Use class-weighted loss, oversample minority classes, augment",
                    details={
                        "imbalance_ratio": imbalance_ratio,
                        "majority": majority_classes,
                        "minority": minority_classes,
                    },
                )
            )

        if minority_classes:
            issues.append(
                QualityIssue(
                    severity=Severity.INFO,
                    category=QualityCategory.IMBALANCE,
                    message=f"Found {len(minority_classes)} minority classes below threshold",
                    location="dataset",
                    suggestion="Apply targeted augmentation for minority classes",
                    details={
                        "minority_classes": minority_classes,
                        "threshold": self._config.rare_class_threshold,
                    },
                )
            )

        passed = len([x for x in issues if x.severity == Severity.ERROR]) == 0

        return ImbalanceAnalysisResult(
            class_distribution=class_counts,
            sorted_distribution=tuple(sorted_dist),
            total_samples=total,
            num_classes=num_classes,
            minority_classes=tuple(minority_classes),
            majority_classes=tuple(majority_classes),
            long_tail_ratio=long_tail_ratio,
            imbalance_ratio=imbalance_ratio,
            recommended_augmentations=recommended_augmentations,
            issues=tuple(issues),
            passed=passed,
        )
