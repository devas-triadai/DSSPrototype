"""Statistics engine for computing comprehensive dataset metrics.

Generates a ``StatisticsReport`` covering:
  - image / annotation / class counts
  - class distribution and imbalance
  - resolution and bounding-box distributions
  - dataset diversity (entropy-based)
  - ontology coverage
  - duplicate ratio
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from backend.dataset_intelligence.interfaces import StatisticsEngineInterface
from backend.dataset_intelligence.models import (
    HarmonizedDataset,
    MergedDataset,
    NormalizedDataset,
    StatisticsReport,
)

logger = logging.getLogger("dss.dataset_intelligence.statistics")


class StatisticsEngine(StatisticsEngineInterface):
    """Compute comprehensive statistics for any post-normalization dataset."""

    def compute(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
    ) -> StatisticsReport:
        logger.info("Statistics computation started | dataset=%s", dataset.dataset_id)
        total_images = len(dataset.images)
        total_annotations = sum(len(img.annotations) for img in dataset.images)
        classes = dataset.classes
        class_count = len(classes)

        objects_per_class: dict[str, int] = defaultdict(int)
        for img in dataset.images:
            for ann in img.annotations:
                cls = ann.ontology_class or ann.normalized_class or ann.class_name
                objects_per_class[cls] += 1

        class_distribution: dict[str, float] = {}
        if total_annotations > 0:
            for cls, count in objects_per_class.items():
                class_distribution[cls] = count / total_annotations

        class_imbalance_ratio = self._compute_imbalance_ratio(objects_per_class)
        average_objects_per_image = total_annotations / total_images if total_images else 0.0

        # Resolution stats
        widths: list[int] = []
        heights: list[int] = []
        resolution_distribution: dict[str, int] = defaultdict(int)
        for img in dataset.images:
            if img.width > 0 and img.height > 0:
                widths.append(img.width)
                heights.append(img.height)
                bucket = f"{img.width}x{img.height}"
                resolution_distribution[bucket] += 1

        avg_width = sum(widths) / len(widths) if widths else 0.0
        avg_height = sum(heights) / len(heights) if heights else 0.0

        # BBox stats
        bbox_widths: list[float] = []
        bbox_heights: list[float] = []
        aspect_ratios: list[float] = []
        bbox_width_distribution: dict[str, int] = defaultdict(int)
        bbox_height_distribution: dict[str, int] = defaultdict(int)
        aspect_ratio_distribution: dict[str, int] = defaultdict(int)

        for img in dataset.images:
            for ann in img.annotations:
                x1, y1, x2, y2 = ann.bbox
                w = x2 - x1
                h = y2 - y1
                bbox_widths.append(w)
                bbox_heights.append(h)
                ar = w / h if h > 0 else 0.0
                aspect_ratios.append(ar)
                bbox_width_distribution[f"{w:.2f}"] += 1
                bbox_height_distribution[f"{h:.2f}"] += 1
                aspect_ratio_distribution[f"{ar:.2f}"] += 1

        # Dataset diversity (Shannon entropy over class distribution)
        diversity = self._compute_diversity(class_distribution)

        # Ontology coverage (fraction of annotations with ontology_class set)
        annotated = 0
        mapped = 0
        for img in dataset.images:
            for ann in img.annotations:
                annotated += 1
                if ann.ontology_class:
                    mapped += 1
        ontology_coverage = mapped / annotated if annotated else 1.0

        # Duplicate ratio (placeholder — would come from DuplicateReport if available)
        duplicate_ratio = 0.0

        report = StatisticsReport(
            dataset_id=dataset.dataset_id,
            total_images=total_images,
            total_annotations=total_annotations,
            classes=sorted(classes),
            class_count=class_count,
            objects_per_class=dict(objects_per_class),
            class_distribution=class_distribution,
            class_imbalance_ratio=class_imbalance_ratio,
            average_objects_per_image=average_objects_per_image,
            average_image_width=avg_width,
            average_image_height=avg_height,
            resolution_distribution=dict(resolution_distribution),
            bbox_width_distribution=dict(bbox_width_distribution),
            bbox_height_distribution=dict(bbox_height_distribution),
            aspect_ratio_distribution=dict(aspect_ratio_distribution),
            ontology_coverage=ontology_coverage,
            duplicate_ratio=duplicate_ratio,
            dataset_diversity=diversity,
        )
        logger.info(
            "Statistics complete | images=%d | annotations=%d | classes=%d | diversity=%.3f",
            total_images,
            total_annotations,
            class_count,
            diversity,
        )
        return report

    @staticmethod
    def _compute_imbalance_ratio(objects_per_class: dict[str, int]) -> float:
        if not objects_per_class:
            return 0.0
        counts = sorted(objects_per_class.values(), reverse=True)
        if len(counts) < 2 or counts[-1] == 0:
            return 0.0
        return counts[0] / counts[-1]

    @staticmethod
    def _compute_diversity(class_distribution: dict[str, float]) -> float:
        if not class_distribution:
            return 0.0
        entropy = 0.0
        for p in class_distribution.values():
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(class_distribution)) if class_distribution else 1.0
        return entropy / max_entropy if max_entropy > 0 else 0.0
