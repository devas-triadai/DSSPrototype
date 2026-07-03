"""Dataset merger — unifies multiple datasets into one while preserving provenance.

Handles:
  - Class ID remapping across datasets
  - Image ID deduplication (prefixes with dataset name)
  - Provenance preservation for every sample
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from backend.dataset_intelligence.interfaces import DatasetMergerInterface
from backend.dataset_intelligence.models import (
    HarmonizedDataset,
    ImageRecord,
    MergedDataset,
    NormalizedDataset,
    ProvenanceRecord,
)

logger = logging.getLogger("dss.dataset_intelligence.merger")


class DatasetMerger(DatasetMergerInterface):
    """Merge multiple datasets into a unified dataset."""

    def merge(
        self,
        datasets: Sequence[NormalizedDataset | HarmonizedDataset],
    ) -> MergedDataset:
        logger.info("Dataset merge started | datasets=%d", len(datasets))
        if not datasets:
            raise ValueError("Cannot merge zero datasets")

        merged_name = "_".join(d.dataset_name for d in datasets)
        merged_id = f"merged_{merged_name}"
        all_images: list[ImageRecord] = []
        all_classes: set[str] = set()
        all_provenance: list[ProvenanceRecord] = []
        source_datasets: list[str] = []

        for ds in datasets:
            source_datasets.append(ds.dataset_id)
            ds_classes = ds.classes
            for cls in ds_classes:
                all_classes.add(cls)

            for img in ds.images:
                # Deduplicate image IDs by prefixing with dataset name
                unique_id = f"{ds.dataset_id}_{img.image_id}"
                anns = []
                for ann in img.annotations:
                    ontology_class = ann.ontology_class or ann.normalized_class or ann.class_name
                    all_classes.add(ontology_class)
                    anns.append(ann.model_copy(update={"ontology_class": ontology_class}))
                prov = img.provenance
                if prov is None:
                    prov = ProvenanceRecord(
                        source_dataset=ds.dataset_id,
                        original_class=img.annotations[0].class_name if img.annotations else "",
                        normalized_class=img.annotations[0].normalized_class
                        if img.annotations
                        else "",
                        ontology_class=img.annotations[0].ontology_class if img.annotations else "",
                        import_format=getattr(ds, "import_format", "unknown"),
                    )
                else:
                    prov = prov.model_copy(update={"source_dataset": ds.dataset_id})
                all_provenance.append(prov)

                all_images.append(
                    ImageRecord(
                        image_id=unique_id,
                        image_path=img.image_path,
                        image_name=img.image_name,
                        width=img.width,
                        height=img.height,
                        channels=img.channels,
                        format=img.format,
                        annotations=anns,
                        checksum=img.checksum,
                        metadata=img.metadata,
                        provenance=prov,
                    )
                )

        result = MergedDataset(
            dataset_id=merged_id,
            dataset_name=merged_name,
            source_datasets=source_datasets,
            images=all_images,
            classes=sorted(all_classes),
            provenance=all_provenance,
        )
        logger.info(
            "Dataset merge complete | merged_id=%s | images=%d | classes=%d",
            merged_id,
            len(result.images),
            len(result.classes),
        )
        return result
