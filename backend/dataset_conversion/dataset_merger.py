from __future__ import annotations

from uuid import uuid4

from backend.dataset_conversion.interfaces import DatasetMergerInterface
from backend.dataset_conversion.models import (
    CanonicalAnnotation,
    CanonicalDataset,
    ImageInfo,
    MergeConfig,
    MergeResult,
)


class DatasetMerger(DatasetMergerInterface):
    async def merge(
        self,
        datasets: list[CanonicalDataset],
        config: MergeConfig | None = None,
    ) -> MergeResult:
        if not datasets:
            raise ValueError("Cannot merge empty list of datasets")

        cfg = config or MergeConfig()
        all_images: dict[str, ImageInfo] = {}
        all_annotations: list[CanonicalAnnotation] = []
        source_names: set[str] = set()
        dedup_count = 0

        for ds in datasets:
            source_names.add(ds.name)
            for img in ds.images:
                img_key = img.id
                if img_key in all_images and cfg.deduplicate_images:
                    dedup_count += 1
                    existing = all_images[img_key]
                    if existing.file_path != img.file_path:
                        new_key = f"{cfg.image_id_prefix}_{img_key}"
                        all_images[new_key] = ImageInfo(
                            id=new_key,
                            file_path=img.file_path,
                            width=img.width,
                            height=img.height,
                            format=img.format,
                            color_space=img.color_space,
                            metadata={**img.metadata, "merged_from": ds.name},
                        )
                        for ann in ds.annotations:
                            if ann.image_id == img_key:
                                all_annotations.append(
                                    self._remap_annotation(
                                        ann,
                                        new_key,
                                        ds,
                                        cfg,
                                    )
                                )
                elif img_key in all_images:
                    new_key = f"{cfg.image_id_prefix}_{img_key}_{uuid4().hex[:4]}"
                    all_images[new_key] = ImageInfo(
                        id=new_key,
                        file_path=img.file_path,
                        width=img.width,
                        height=img.height,
                        format=img.format,
                        color_space=img.color_space,
                        metadata={**img.metadata, "merged_from": ds.name},
                    )
                    for ann in ds.annotations:
                        if ann.image_id == img_key:
                            all_annotations.append(self._remap_annotation(ann, new_key, ds, cfg))
                else:
                    all_images[img_key] = img
                    for ann in ds.annotations:
                        if ann.image_id == img_key:
                            all_annotations.append(self._remap_annotation(ann, img_key, ds, cfg))

        merged_id = f"merged_{uuid4().hex[:8]}"
        merged = CanonicalDataset(
            id=merged_id,
            name=cfg.image_id_prefix,
            images=tuple(all_images.values()),
            annotations=tuple(all_annotations),
            image_count=len(all_images),
            annotation_count=len(all_annotations),
            class_count=len({a.canonical_label for a in all_annotations}),
            source_datasets=tuple(sorted(source_names)),
        )

        return MergeResult(
            dataset=merged,
            total_images=merged.image_count,
            total_annotations=merged.annotation_count,
            deduplicated_count=dedup_count,
            source_datasets=tuple(sorted(source_names)),
        )

    def _remap_annotation(
        self,
        ann: CanonicalAnnotation,
        new_image_id: str,
        ds: CanonicalDataset,
        cfg: MergeConfig,
    ) -> CanonicalAnnotation:
        return CanonicalAnnotation(
            id=f"{cfg.annotation_id_prefix}_{ann.id}_{uuid4().hex[:4]}",
            image_id=new_image_id,
            canonical_label=ann.canonical_label,
            canonical_name=ann.canonical_name,
            geometry_type=ann.geometry_type,
            x=ann.x,
            y=ann.y,
            width=ann.width,
            height=ann.height,
            confidence=ann.confidence,
            source_annotation_id=ann.source_annotation_id,
            source_label=ann.source_label,
            metadata={**ann.metadata, "merged_from": ds.name},
        )
