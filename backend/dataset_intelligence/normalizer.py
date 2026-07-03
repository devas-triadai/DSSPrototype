"""Dataset normalizer producing a canonical ``NormalizedDataset``.

Normalizes:
  - class names (lowercase, snake_case, strip whitespace)
  - file naming conventions (kebab-case stem, lowercase extension)
  - bounding boxes to ``xyxy_normalized``
  - directory structure metadata
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from backend.dataset_intelligence.interfaces import DatasetNormalizerInterface
from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
    RawDataset,
)

logger = logging.getLogger("dss.dataset_intelligence.normalizer")


class DatasetNormalizer(DatasetNormalizerInterface):
    """Normalize a raw dataset into a canonical representation.

    All transformations are pure functions; the original ``RawDataset`` is
    never mutated.
    """

    @staticmethod
    def _normalize_class_name(name: str) -> str:
        """Convert any class name to snake_case."""
        s = name.strip().lower()
        s = re.sub(r"[^\w\s]+", "", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s.strip("_")

    @staticmethod
    def _normalize_filename(name: str) -> str:
        """Normalize a file name to lowercase with standard extension."""
        p = Path(name)
        ext = p.suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            ext = ".jpg"
        stem = re.sub(r"[^\w]+", "-", p.stem).lower().strip("-")
        return f"{stem}{ext}"

    def normalize(self, dataset: RawDataset) -> NormalizedDataset:
        logger.info("Normalization started | dataset=%s", dataset.dataset_id)
        normalization_log: list[str] = []
        class_mapping: dict[str, str] = {}
        normalized_classes: set[str] = set()

        # Build class mapping
        for cls in dataset.classes:
            norm = self._normalize_class_name(cls)
            class_mapping[cls] = norm
            normalized_classes.add(norm)

        normalized_images: list[ImageRecord] = []
        for img in dataset.images:
            norm_name = self._normalize_filename(img.image_name)
            if norm_name != img.image_name:
                normalization_log.append(f"Renamed {img.image_name} -> {norm_name}")

            norm_annotations: list[Annotation] = []
            for ann in img.annotations:
                norm_class = class_mapping.get(ann.class_name, ann.class_name)
                if norm_class != ann.class_name:
                    normalization_log.append(f"Class remap: {ann.class_name} -> {norm_class}")

                x1, y1, x2, y2 = ann.bbox
                # Ensure normalized xyxy
                norm_bbox = (
                    max(0.0, min(x1, 1.0)),
                    max(0.0, min(y1, 1.0)),
                    max(0.0, min(x2, 1.0)),
                    max(0.0, min(y2, 1.0)),
                )
                if norm_bbox != ann.bbox:
                    normalization_log.append(
                        f"Clamped bbox for {img.image_id}: {ann.bbox} -> {norm_bbox}"
                    )

                norm_annotations.append(
                    Annotation(
                        class_name=ann.class_name,
                        normalized_class=norm_class,
                        ontology_class=ann.ontology_class,
                        bbox=norm_bbox,
                        bbox_format="xyxy_normalized",
                        confidence=ann.confidence,
                        segmentation=ann.segmentation,
                        attributes=ann.attributes,
                    )
                )

            norm_provenance = None
            if img.provenance is not None:
                norm_provenance = (
                    img.provenance.model_copy(update={"normalized_class": norm_class})
                    if norm_annotations
                    else img.provenance
                )

            normalized_images.append(
                ImageRecord(
                    image_id=img.image_id,
                    image_path=img.image_path,
                    image_name=norm_name,
                    width=img.width,
                    height=img.height,
                    channels=img.channels,
                    format=img.format,
                    annotations=norm_annotations,
                    checksum=img.checksum,
                    metadata=img.metadata,
                    provenance=norm_provenance,
                )
            )

        result = NormalizedDataset(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.dataset_name,
            images=normalized_images,
            classes=sorted(normalized_classes),
            class_mapping=class_mapping,
            metadata=dataset.metadata,
            normalization_log=normalization_log,
        )
        logger.info(
            "Normalization complete | dataset=%s | classes=%d | images=%d | log_entries=%d",
            dataset.dataset_id,
            len(result.classes),
            len(result.images),
            len(normalization_log),
        )
        return result
