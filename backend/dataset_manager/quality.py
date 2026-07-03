"""Quality engine — assesses dataset quality and produces a QualityReport.

Computes a Quality Score based on:
  - Annotation completeness (what fraction of images have annotations)
  - Dataset balance (inverse of class imbalance ratio, capped at 1.0)
  - Image quality metadata (placeholder — extend with real metrics)
  - Duplicate percentage (inverse: fewer duplicates is better)
  - Missing labels (inverse: fewer missing is better)
  - Validation score (fraction of validation checks passed)
"""

import logging

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import QualityEngineInterface
from backend.dataset_manager.models import DatasetQuality, DatasetStatistics, DatasetValidation

logger = logging.getLogger("dss.dataset_manager.quality")


class QualityEngine(QualityEngineInterface):
    """Assesses dataset quality using multiple weighted factors."""

    def __init__(self) -> None:
        self._config = dm_config

    def assess(
        self,
        statistics: DatasetStatistics,
        validation: DatasetValidation,
    ) -> DatasetQuality:
        logger.info("Quality assessment started: %s", statistics.dataset_id)

        annotation_completeness = statistics.dataset_completeness

        dataset_balance = self._compute_balance(statistics.class_imbalance_ratio)

        image_quality_metadata = 1.0

        duplicate_pct = self._compute_duplicate_pct(validation)

        missing_labels = len(validation.missing_labels)

        validation_score = (
            validation.passed_checks / validation.total_checks
            if validation.total_checks > 0
            else 0.0
        )

        quality_score = self._compute_score(
            annotation_completeness=annotation_completeness,
            dataset_balance=dataset_balance,
            image_quality_metadata=image_quality_metadata,
            duplicate_pct=duplicate_pct,
            missing_labels=missing_labels,
            validation_score=validation_score,
            total_images=statistics.total_images,
        )

        warnings: list[str] = []
        errors: list[str] = []

        if duplicate_pct > self._config.quality_max_duplicate_pct:
            warnings.append(
                f"Duplicate percentage ({duplicate_pct:.1f}%) exceeds "
                f"threshold ({self._config.quality_max_duplicate_pct:.1f}%)",
            )
        if missing_labels > 0:
            warnings.append(f"{missing_labels} missing label(s) detected")
        if len(statistics.classes) < 1:
            errors.append("No classes detected in dataset")
        if validation.failed_checks > 0:
            errors.append(f"{validation.failed_checks} validation check(s) failed")

        quality = DatasetQuality(
            dataset_id=statistics.dataset_id,
            quality_score=round(quality_score, 4),
            annotation_completeness=round(annotation_completeness, 4),
            dataset_balance=round(dataset_balance, 4),
            image_quality_metadata=round(image_quality_metadata, 4),
            duplicate_percentage=round(duplicate_pct, 4),
            missing_labels=missing_labels,
            validation_score=round(validation_score, 4),
            warnings=warnings,
            errors=errors,
        )

        logger.info(
            "Quality assessment completed: score=%.4f, warnings=%d, errors=%d",
            quality_score,
            len(warnings),
            len(errors),
        )
        return quality

    def _compute_balance(self, imbalance_ratio: float) -> float:
        if imbalance_ratio == 0.0:
            return 1.0
        if imbalance_ratio == float("inf"):
            return 0.0
        balance = 1.0 / imbalance_ratio
        return min(balance, 1.0)

    def _compute_duplicate_pct(self, validation: DatasetValidation) -> float:
        total_dup = (
            len(validation.duplicate_images)
            + len(validation.duplicate_annotations)
        )
        return float(total_dup)

    def _compute_score(
        self,
        annotation_completeness: float,
        dataset_balance: float,
        image_quality_metadata: float,
        duplicate_pct: float,
        missing_labels: int,
        validation_score: float,
        total_images: int,
    ) -> float:
        w_completeness = 0.25
        w_balance = 0.15
        w_image_quality = 0.10
        w_duplicate = 0.15
        w_missing = 0.10
        w_validation = 0.25

        duplicate_factor = max(0.0, 1.0 - (duplicate_pct / 100.0))
        missing_factor = max(
            0.0,
            1.0 - (missing_labels / max(total_images, 1)),
        )

        score = (
            w_completeness * annotation_completeness
            + w_balance * dataset_balance
            + w_image_quality * image_quality_metadata
            + w_duplicate * duplicate_factor
            + w_missing * missing_factor
            + w_validation * validation_score
        )

        return max(0.0, min(score, 1.0))
