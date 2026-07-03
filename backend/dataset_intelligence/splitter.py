"""Dataset splitter — train / validation / test stratified splitting.

Operates on image IDs. When ``stratified=True``, preserves class
distribution across splits.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from backend.dataset_intelligence.interfaces import DatasetSplitterInterface
from backend.dataset_intelligence.models import (
    HarmonizedDataset,
    MergedDataset,
    NormalizedDataset,
)

logger = logging.getLogger("dss.dataset_intelligence.splitter")


class DatasetSplitter(DatasetSplitterInterface):
    """Split a dataset into train/validation/test sets."""

    def split(
        self,
        dataset: NormalizedDataset | HarmonizedDataset | MergedDataset,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = True,
    ) -> dict[str, list[str]]:
        logger.info(
            "Split started | dataset=%s | train=%.2f | val=%.2f | test=%.2f | stratified=%s",
            dataset.dataset_id,
            train_ratio,
            validation_ratio,
            test_ratio,
            stratified,
        )
        if abs(train_ratio + validation_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Ratios must sum to 1.0")

        image_ids = [img.image_id for img in dataset.images]
        if not image_ids:
            return {"train": [], "validation": [], "test": []}

        rng = random.Random(seed)

        if stratified and dataset.classes:
            # Group images by their dominant class
            class_to_images: dict[str, list[str]] = defaultdict(list)
            for img in dataset.images:
                if img.annotations:
                    dominant = (
                        img.annotations[0].ontology_class
                        or img.annotations[0].normalized_class
                        or img.annotations[0].class_name
                    )
                else:
                    dominant = "__background__"
                class_to_images[dominant].append(img.image_id)

            train_ids: list[str] = []
            val_ids: list[str] = []
            test_ids: list[str] = []

            for cls, ids in class_to_images.items():
                rng.shuffle(ids)
                n = len(ids)
                n_train = int(n * train_ratio)
                n_val = int(n * validation_ratio)
                # Ensure at least one sample per split if possible
                if n_train == 0 and n > 0:
                    n_train = 1
                if n_val == 0 and n > 1:
                    n_val = 1
                train_ids.extend(ids[:n_train])
                val_ids.extend(ids[n_train : n_train + n_val])
                test_ids.extend(ids[n_train + n_val :])

            rng.shuffle(train_ids)
            rng.shuffle(val_ids)
            rng.shuffle(test_ids)
        else:
            rng.shuffle(image_ids)
            n = len(image_ids)
            n_train = int(n * train_ratio)
            n_val = int(n * validation_ratio)
            train_ids = image_ids[:n_train]
            val_ids = image_ids[n_train : n_train + n_val]
            test_ids = image_ids[n_train + n_val :]

        splits = {
            "train": train_ids,
            "validation": val_ids,
            "test": test_ids,
        }
        logger.info(
            "Split complete | train=%d | val=%d | test=%d",
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
        return splits
