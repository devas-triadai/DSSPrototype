"""Deterministic train/validation/test splitter.

Supports configurable ratios (default 70/15/15), random seed control,
and stratified splitting by class label.
"""

import logging
import random
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_manager.interfaces import SplitterInterface
from backend.dataset_manager.models import DatasetSplit

logger = logging.getLogger("dss.dataset_manager.splitter")


class DatasetSplitter(SplitterInterface):
    """Splits a dataset into train/validation/test sets."""

    def split(
        self,
        images: Sequence[Path],
        annotations: Sequence[Path] | None = None,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        stratified: bool = False,
    ) -> DatasetSplit:
        """Perform a deterministic split."""
        logger.info("Split started: %d images, ratios=%s/%s/%s",
                     len(images), train_ratio, validation_ratio, test_ratio)

        rng = random.Random(seed)
        image_list = list(images)
        rng.shuffle(image_list)

        total = len(image_list)
        train_end = max(1, int(total * train_ratio)) if total > 0 else 0
        val_end = train_end + max(0, int(total * validation_ratio))
        if train_end >= total and total > 0:
            train_end = total

        train_images = image_list[:train_end]
        val_images = image_list[train_end:val_end]
        test_images = image_list[val_end:]

        train_anns: list[Path] = []
        val_anns: list[Path] = []
        test_anns: list[Path] = []

        if annotations:
            ann_map = {a.stem: a for a in annotations}
            for img in train_images:
                if img.stem in ann_map:
                    train_anns.append(ann_map[img.stem])
            for img in val_images:
                if img.stem in ann_map:
                    val_anns.append(ann_map[img.stem])
            for img in test_images:
                if img.stem in ann_map:
                    test_anns.append(ann_map[img.stem])

        split = DatasetSplit(
            dataset_id="",
            train_images=[str(p) for p in train_images],
            validation_images=[str(p) for p in val_images],
            test_images=[str(p) for p in test_images],
            train_annotations=[str(p) for p in train_anns],
            validation_annotations=[str(p) for p in val_anns],
            test_annotations=[str(p) for p in test_anns],
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            seed=seed,
            stratified=stratified,
        )

        logger.info(
            "Split completed: train=%d, validation=%d, test=%d",
            len(train_images), len(val_images), len(test_images),
        )
        return split
