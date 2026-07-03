"""Quality engine — computes a composite quality score and gate report.

Factors:
  - duplicate ratio
  - class imbalance
  - missing annotations / images
  - annotation consistency (bbox validity)
  - ontology coverage

Compares against thresholds defined in ``DatasetIntelligenceConfig``.
"""

from __future__ import annotations

import logging

from backend.dataset_intelligence.config import di_config
from backend.dataset_intelligence.interfaces import QualityEngineInterface
from backend.dataset_intelligence.models import (
    DuplicateReport,
    HarmonizedDataset,
    MergedDataset,
    NormalizedDataset,
    QualityReport,
    StatisticsReport,
    ValidationReport,
)

logger = logging.getLogger("dss.dataset_intelligence.quality")


class QualityEngine(QualityEngineInterface):
    """Assess dataset quality and produce a gating report."""

    def assess(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        validation: ValidationReport | None,
        duplicates: DuplicateReport | None,
        statistics: StatisticsReport | None,
    ) -> QualityReport:
        logger.info("Quality assessment started | dataset=%s", dataset.dataset_id)
        warnings: list[str] = []
        errors: list[str] = []
        scores: list[float] = []

        # Duplicate score
        dup_ratio = duplicates.duplicate_ratio if duplicates else 0.0
        dup_score = max(0.0, 1.0 - dup_ratio / di_config.max_duplicate_ratio)
        scores.append(dup_score)
        if dup_ratio > di_config.max_duplicate_ratio:
            errors.append(
                f"Duplicate ratio {dup_ratio:.2f} exceeds threshold {di_config.max_duplicate_ratio}"
            )

        # Class imbalance score
        imbalance = statistics.class_imbalance_ratio if statistics else 0.0
        imbalance_score = max(0.0, 1.0 - imbalance / di_config.max_class_imbalance_ratio)
        scores.append(imbalance_score)
        if imbalance > di_config.max_class_imbalance_ratio:
            warnings.append(
                f"Class imbalance {imbalance:.2f} exceeds "
                f"threshold {di_config.max_class_imbalance_ratio}"
            )

        # Missing annotations / images
        missing_images = len(validation.missing_images) if validation else 0
        missing_anns = len(validation.missing_annotations) if validation else 0
        empty_anns = len(validation.empty_annotations) if validation else 0
        total_images = len(dataset.images)
        missing_score = 1.0
        if total_images > 0:
            missing_ratio = (missing_images + missing_anns + empty_anns) / total_images
            missing_score = max(0.0, 1.0 - missing_ratio)
        scores.append(missing_score)
        if missing_images > 0:
            warnings.append(f"Missing images: {missing_images}")
        if empty_anns > 0:
            warnings.append(f"Empty annotations: {empty_anns}")

        # Annotation consistency (bbox validity)
        invalid_bbox = len(validation.invalid_bounding_boxes) if validation else 0
        negative_coords = len(validation.negative_coordinates) if validation else 0
        total_anns = sum(len(img.annotations) for img in dataset.images)
        consistency_score = 1.0
        if total_anns > 0:
            consistency_score = max(0.0, 1.0 - (invalid_bbox + negative_coords) / total_anns)
        scores.append(consistency_score)
        if invalid_bbox > 0:
            warnings.append(f"Invalid bounding boxes: {invalid_bbox}")
        if negative_coords > 0:
            warnings.append(f"Negative coordinates: {negative_coords}")

        # Ontology coverage
        coverage = statistics.ontology_coverage if statistics else 0.0
        coverage_score = min(1.0, coverage / di_config.min_ontology_coverage)
        scores.append(coverage_score)
        if coverage < di_config.min_ontology_coverage:
            warnings.append(
                f"Ontology coverage {coverage:.2f} below "
                f"threshold {di_config.min_ontology_coverage}"
            )

        # Volume checks
        if total_images < di_config.min_image_count:
            errors.append(f"Image count {total_images} below minimum {di_config.min_image_count}")
        if total_anns < di_config.min_annotation_count:
            errors.append(
                f"Annotation count {total_anns} below minimum {di_config.min_annotation_count}"
            )
        if len(dataset.classes) < di_config.min_class_count:
            errors.append(
                f"Class count {len(dataset.classes)} below minimum {di_config.min_class_count}"
            )

        quality_score = sum(scores) / len(scores) if scores else 0.0
        passed = quality_score >= di_config.min_quality_score and len(errors) == 0

        report = QualityReport(
            dataset_id=dataset.dataset_id,
            quality_score=quality_score,
            duplicate_ratio=dup_ratio,
            class_imbalance_ratio=imbalance,
            missing_annotations=missing_anns + empty_anns,
            missing_images=missing_images,
            annotation_consistency=consistency_score,
            ontology_coverage=coverage,
            warnings=warnings,
            errors=errors,
            passed=passed,
        )
        logger.info(
            "Quality assessment complete | score=%.3f | passed=%s | warnings=%d | errors=%d",
            quality_score,
            passed,
            len(warnings),
            len(errors),
        )
        return report
