"""Statistics engine — computes comprehensive dataset statistics.

Generates:
  - Total images / annotations
  - Classes and objects per class
  - Class distribution and imbalance ratio
  - Average objects per image
  - Average image size
  - Resolution, bbox-size, and aspect-ratio distributions
  - Dataset completeness and coverage score
"""

import json
import logging
import statistics
from pathlib import Path

from backend.dataset_manager.config import dm_config
from backend.dataset_manager.interfaces import StatisticsEngineInterface
from backend.dataset_manager.models import DatasetStatistics

logger = logging.getLogger("dss.dataset_manager.statistics")


class StatisticsEngine(StatisticsEngineInterface):
    """Computes production-grade statistics for CV datasets."""

    def __init__(self) -> None:
        self._config = dm_config

    def compute(
        self,
        dataset_path: Path,
        annotation_path: Path | None = None,
    ) -> DatasetStatistics:
        """Compute comprehensive statistics."""
        logger.info("Statistics generation started: %s", dataset_path)
        images = self._find_images(dataset_path)
        annotations = self._find_annotations(annotation_path or dataset_path)

        total_images = len(images)
        image_sizes: list[tuple[int, int]] = self._get_image_sizes(images)
        class_counts: dict[str, int] = self._count_classes(annotations)
        total_annotations = sum(class_counts.values())
        classes = sorted(class_counts.keys())

        class_distribution = {
            cls: count / total_annotations if total_annotations > 0 else 0.0
            for cls, count in class_counts.items()
        }

        class_imbalance_ratio = self._compute_imbalance(class_counts)

        avg_objects = (
            total_annotations / total_images if total_images > 0 else 0.0
        )

        avg_w = statistics.mean([w for w, _ in image_sizes]) if image_sizes else 0.0
        avg_h = statistics.mean([h for _, h in image_sizes]) if image_sizes else 0.0

        res_dist = self._resolution_distribution(image_sizes)
        bbox_w, bbox_h, aspect = self._bbox_statistics(annotations)
        comp = self._completeness(images, annotations)
        coverage = self._coverage(class_distribution)

        stats = DatasetStatistics(
            dataset_id=dataset_path.name,
            total_images=total_images,
            total_annotations=total_annotations,
            classes=classes,
            objects_per_class=class_counts,
            class_distribution=class_distribution,
            class_imbalance_ratio=class_imbalance_ratio,
            average_objects_per_image=avg_objects,
            average_image_width=avg_w,
            average_image_height=avg_h,
            resolution_distribution=res_dist,
            bbox_width_distribution=bbox_w,
            bbox_height_distribution=bbox_h,
            aspect_ratio_distribution=aspect,
            dataset_completeness=comp,
            class_count=len(classes),
            coverage_score=coverage,
        )

        logger.info(
            "Statistics generated: %d images, %d annotations, %d classes",
            total_images,
            total_annotations,
            len(classes),
        )
        return stats

    def _find_images(self, path: Path) -> list[Path]:
        exts = self._config.supported_image_extensions
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]

    def _find_annotations(self, path: Path) -> list[Path]:
        exts = self._config.supported_annotation_extensions
        return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in exts]

    def _get_image_sizes(self, images: list[Path]) -> list[tuple[int, int]]:
        sizes: list[tuple[int, int]] = []
        for img in images:
            try:
                from PIL import Image
                with Image.open(img) as im:
                    sizes.append(im.size)
            except Exception:
                pass
        return sizes

    def _count_classes(self, annotations: list[Path]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ann in annotations:
            try:
                data = json.loads(ann.read_text())
                self._accumulate_classes(data, counts)
            except Exception:
                pass
        return counts

    def _accumulate_classes(self, data: object, counts: dict[str, int]) -> None:
        if isinstance(data, dict):
            for ann in data.get("annotations", []):
                cat_id = ann.get("category_id")
                if cat_id is not None:
                    key = f"class_{cat_id}"
                    counts[key] = counts.get(key, 0) + 1

    def _compute_imbalance(self, counts: dict[str, int]) -> float:
        if not counts:
            return 0.0
        vals = list(counts.values())
        return max(vals) / min(vals) if min(vals) > 0 else float("inf")

    def _resolution_distribution(
        self, sizes: list[tuple[int, int]],
    ) -> dict[str, int]:
        dist: dict[str, int] = {}
        for w, h in sizes:
            if w >= 1920 and h >= 1080:
                key = "hd"
            elif w >= 1280 and h >= 720:
                key = "720p"
            elif w >= 640 and h >= 480:
                key = "vga"
            else:
                key = "low"
            dist[key] = dist.get(key, 0) + 1
        return dist

    def _bbox_statistics(
        self, annotations: list[Path],
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        w_dist: dict[str, int] = {}
        h_dist: dict[str, int] = {}
        a_dist: dict[str, int] = {}
        for ann in annotations:
            try:
                data = json.loads(ann.read_text())
                for a in data.get("annotations", []):
                    bbox = a.get("bbox")
                    if bbox and len(bbox) == 4:
                        w, h = bbox[2], bbox[3]
                        w_dist[self._size_bin(w)] = w_dist.get(self._size_bin(w), 0) + 1
                        h_dist[self._size_bin(h)] = h_dist.get(self._size_bin(h), 0) + 1
                        ar = w / h if h > 0 else 1.0
                        a_dist[self._aspect_bin(ar)] = a_dist.get(self._aspect_bin(ar), 0) + 1
            except Exception:
                pass
        return w_dist, h_dist, a_dist

    def _size_bin(self, val: float) -> str:
        if val < 32:
            return "tiny"
        elif val < 64:
            return "small"
        elif val < 128:
            return "medium"
        elif val < 256:
            return "large"
        return "xlarge"

    def _aspect_bin(self, ar: float) -> str:
        if ar < 0.5:
            return "tall"
        elif ar < 1.5:
            return "square"
        elif ar < 3.0:
            return "wide"
        return "xwide"

    def _completeness(self, images: list[Path], annotations: list[Path]) -> float:
        if not images:
            return 1.0
        img_stems = {i.stem for i in images}
        ann_stems = {a.stem for a in annotations}
        annotated = len(img_stems & ann_stems)
        return annotated / len(images)

    def _coverage(self, distribution: dict[str, float]) -> float:
        if not distribution:
            return 0.0
        vals = list(distribution.values())
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return 1.0 - std
