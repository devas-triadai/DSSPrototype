from __future__ import annotations

import random

from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    ImageInfo,
    SplitConfig,
    SplitResult,
    SplitStrategy,
    SplitType,
)


class DatasetSplitter:
    async def split(
        self,
        dataset: CanonicalDataset,
        config: SplitConfig | None = None,
    ) -> SplitResult:
        cfg = config or SplitConfig()

        if cfg.train_ratio + cfg.val_ratio + cfg.test_ratio > 1.0:
            raise ValueError(
                f"Split ratios sum to {cfg.train_ratio + cfg.val_ratio + cfg.test_ratio}, "
                f"expected <= 1.0"
            )

        rng = random.Random(cfg.seed)
        images = list(dataset.images)
        if cfg.shuffle:
            rng.shuffle(images)

        if cfg.strategy == SplitStrategy.RANDOM:
            return self._random_split(images, dataset.annotations, cfg, rng)
        elif cfg.strategy == SplitStrategy.STRATIFIED:
            return self._stratified_split(images, dataset.annotations, cfg, rng)
        elif cfg.strategy == SplitStrategy.CLASS_BALANCED:
            return self._class_balanced_split(images, dataset.annotations, cfg, rng)
        else:
            raise ValueError(f"Unknown split strategy: {cfg.strategy}")

    def _random_split(
        self,
        images: list[ImageInfo],
        annotations: tuple[CanonicalAnnotation, ...],
        cfg: SplitConfig,
        rng: random.Random,
    ) -> SplitResult:
        n = len(images)
        train_end = int(n * cfg.train_ratio)
        val_end = train_end + int(n * cfg.val_ratio)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        ann_by_img: dict[str, list[CanonicalAnnotation]] = {}
        for ann in annotations:
            ann_by_img.setdefault(ann.image_id, []).append(ann)

        train_ds = self._build_split(f"{cfg.seed}_train", SplitType.TRAIN, train_images, ann_by_img)
        val_ds = self._build_split(f"{cfg.seed}_val", SplitType.VAL, val_images, ann_by_img)
        test_ds = self._build_split(f"{cfg.seed}_test", SplitType.TEST, test_images, ann_by_img)

        actual_train = len(train_images) / n if n > 0 else 0
        actual_val = len(val_images) / n if n > 0 else 0
        actual_test = len(test_images) / n if n > 0 else 0

        return SplitResult(
            train=train_ds,
            val=val_ds,
            test=test_ds,
            train_ratio=actual_train,
            val_ratio=actual_val,
            test_ratio=actual_test,
        )

    def _stratified_split(
        self,
        images: list[ImageInfo],
        annotations: tuple[CanonicalAnnotation, ...],
        cfg: SplitConfig,
        rng: random.Random,
    ) -> SplitResult:
        ann_by_img: dict[str, list[CanonicalAnnotation]] = {}
        for ann in annotations:
            ann_by_img.setdefault(ann.image_id, []).append(ann)

        label_counts: dict[str, int] = {}
        for ann in annotations:
            label_counts[ann.canonical_label] = label_counts.get(ann.canonical_label, 0) + 1

        sorted_labels = sorted(label_counts.keys(), key=lambda x: -label_counts[x])

        train_images: list[ImageInfo] = []
        val_images: list[ImageInfo] = []
        test_images: list[ImageInfo] = []

        n = len(images)
        train_target = int(n * cfg.train_ratio)
        val_target = int(n * cfg.val_ratio)

        for label in sorted_labels:
            label_imgs = [
                img
                for img in images
                if img.id in ann_by_img
                and any(a.canonical_label == label for a in ann_by_img[img.id])
            ]
            rng.shuffle(label_imgs)
            t = int(len(label_imgs) * cfg.train_ratio)
            v = int(len(label_imgs) * cfg.val_ratio)
            train_images.extend(label_imgs[:t])
            val_images.extend(label_imgs[t : t + v])
            test_images.extend(label_imgs[t + v :])

        unlabeled = [img for img in images if img.id not in ann_by_img]
        rng.shuffle(unlabeled)
        t2 = int(len(unlabeled) * cfg.train_ratio)
        v2 = int(len(unlabeled) * cfg.val_ratio)
        train_images.extend(unlabeled[:t2])
        val_images.extend(unlabeled[t2 : t2 + v2])
        test_images.extend(unlabeled[t2 + v2 :])

        if len(train_images) > train_target:
            train_images = train_images[:train_target]
        if len(val_images) > val_target:
            val_images = val_images[:val_target]

        train_ds = self._build_split(f"{cfg.seed}_train", SplitType.TRAIN, train_images, ann_by_img)
        val_ds = self._build_split(f"{cfg.seed}_val", SplitType.VAL, val_images, ann_by_img)
        test_ds = self._build_split(f"{cfg.seed}_test", SplitType.TEST, test_images, ann_by_img)

        return SplitResult(
            train=train_ds,
            val=val_ds,
            test=test_ds,
            train_ratio=len(train_images) / n if n else 0,
            val_ratio=len(val_images) / n if n else 0,
            test_ratio=len(test_images) / n if n else 0,
        )

    def _class_balanced_split(
        self,
        images: list[ImageInfo],
        annotations: tuple[CanonicalAnnotation, ...],
        cfg: SplitConfig,
        rng: random.Random,
    ) -> SplitResult:
        return self._stratified_split(images, annotations, cfg, rng)

    def _build_split(
        self,
        name_suffix: str,
        split_type: SplitType,
        split_images: list[ImageInfo],
        ann_by_img: dict[str, list[CanonicalAnnotation]],
    ) -> CanonicalDataset:
        split_anns: list[CanonicalAnnotation] = []
        img_set = {img.id for img in split_images}
        for img_id in img_set:
            split_anns.extend(ann_by_img.get(img_id, []))

        unique_labels = {a.canonical_label for a in split_anns}

        return CanonicalDataset(
            id=f"{split_type.value}_{name_suffix}",
            name=f"dataset_{split_type.value}",
            images=tuple(split_images),
            annotations=tuple(split_anns),
            image_count=len(split_images),
            annotation_count=len(split_anns),
            class_count=len(unique_labels),
        )
