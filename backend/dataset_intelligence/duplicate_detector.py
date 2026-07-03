"""Duplicate detection engine.

Detects duplicates across multiple dimensions:
  - Identical filenames
  - Identical SHA256 hashes
  - Identical metadata dictionaries
  - Near-duplicate images (perceptual hash architecture, placeholder impl)
  - Duplicate annotations (same bbox + class on same image)
  - Duplicate objects (same bbox + class across different images)

Produces a ``DuplicateReport`` with full provenance for every duplicate pair.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

from backend.dataset_intelligence.interfaces import DuplicateDetectorInterface
from backend.dataset_intelligence.models import (
    DuplicateEntry,
    DuplicateReport,
    NormalizedDataset,
)

logger = logging.getLogger("dss.dataset_intelligence.duplicate_detector")


class DuplicateDetector(DuplicateDetectorInterface):
    """Detect duplicates in a normalized dataset."""

    def detect(self, dataset: NormalizedDataset) -> DuplicateReport:
        logger.info("Duplicate detection started | dataset=%s", dataset.dataset_id)
        duplicates: list[DuplicateEntry] = []

        # Filename duplicates
        filename_groups: dict[str, list[str]] = defaultdict(list)
        for img in dataset.images:
            filename_groups[img.image_name].append(img.image_id)
        for name, ids in filename_groups.items():
            if len(ids) > 1:
                for dup_id in ids[1:]:
                    duplicates.append(
                        DuplicateEntry(
                            duplicate_type="filename",
                            original_id=ids[0],
                            duplicate_id=dup_id,
                            similarity_score=1.0,
                            reason=f"Duplicate filename: {name}",
                        )
                    )

        # Hash duplicates
        hash_groups: dict[str, list[str]] = defaultdict(list)
        for img in dataset.images:
            if img.checksum:
                hash_groups[img.checksum].append(img.image_id)
        for h, ids in hash_groups.items():
            if len(ids) > 1:
                for dup_id in ids[1:]:
                    duplicates.append(
                        DuplicateEntry(
                            duplicate_type="hash",
                            original_id=ids[0],
                            duplicate_id=dup_id,
                            similarity_score=1.0,
                            reason=f"Identical SHA256: {h[:16]}...",
                        )
                    )

        # Metadata duplicates
        metadata_groups: dict[str, list[str]] = defaultdict(list)
        for img in dataset.images:
            key = hashlib.sha256(str(sorted(img.metadata.items())).encode()).hexdigest()
            metadata_groups[key].append(img.image_id)
        for key, ids in metadata_groups.items():
            if len(ids) > 1:
                for dup_id in ids[1:]:
                    duplicates.append(
                        DuplicateEntry(
                            duplicate_type="metadata",
                            original_id=ids[0],
                            duplicate_id=dup_id,
                            similarity_score=1.0,
                            reason="Identical metadata",
                        )
                    )

        # Near-duplicate images (architecture placeholder)
        # In production this would compare perceptual hashes (e.g., imagehash)
        near_duplicates = self._detect_near_duplicates(dataset)
        duplicates.extend(near_duplicates)

        # Duplicate annotations (same image, same class, same bbox)
        for img in dataset.images:
            seen_anns: dict[str, str] = {}
            for ann in img.annotations:
                ann_key = f"{ann.normalized_class or ann.class_name}:{ann.bbox}"
                if ann_key in seen_anns:
                    duplicates.append(
                        DuplicateEntry(
                            duplicate_type="annotation",
                            original_id=img.image_id,
                            duplicate_id=img.image_id,
                            similarity_score=1.0,
                            reason=f"Duplicate annotation on {img.image_id}: {ann_key}",
                        )
                    )
                else:
                    seen_anns[ann_key] = img.image_id

        # Duplicate objects across images (same class, same bbox within tolerance)
        object_groups: dict[str, list[str]] = defaultdict(list)
        for img in dataset.images:
            for ann in img.annotations:
                key = f"{ann.normalized_class or ann.class_name}:{ann.bbox}"
                object_groups[key].append(img.image_id)
        for key, ids in object_groups.items():
            if len(ids) > 1:
                for dup_id in ids[1:]:
                    duplicates.append(
                        DuplicateEntry(
                            duplicate_type="object",
                            original_id=ids[0],
                            duplicate_id=dup_id,
                            similarity_score=1.0,
                            reason=f"Duplicate object across images: {key}",
                        )
                    )

        unique_dup_images = len(
            {
                d.duplicate_id
                for d in duplicates
                if d.duplicate_type in {"filename", "hash", "metadata", "near_duplicate"}
            }
        )
        unique_dup_anns = len(
            {d.duplicate_id for d in duplicates if d.duplicate_type in {"annotation", "object"}}
        )
        dup_ratio = unique_dup_images / len(dataset.images) if dataset.images else 0.0

        report = DuplicateReport(
            dataset_id=dataset.dataset_id,
            duplicates=duplicates,
            duplicate_image_count=unique_dup_images,
            duplicate_annotation_count=unique_dup_anns,
            duplicate_ratio=dup_ratio,
        )
        logger.info(
            "Duplicate detection complete | dataset=%s | image_dups=%d | ann_dups=%d | ratio=%.3f",
            dataset.dataset_id,
            unique_dup_images,
            unique_dup_anns,
            dup_ratio,
        )
        return report

    def _detect_near_duplicates(self, dataset: NormalizedDataset) -> list[DuplicateEntry]:
        """Architecture placeholder for near-duplicate image detection.

        In production, compute perceptual hashes (e.g., phash, dhash) and
        compare with Hamming distance threshold.
        """
        # Placeholder: no near-duplicate detection without imagehash library
        return []
