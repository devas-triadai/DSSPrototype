from __future__ import annotations

from collections.abc import Sequence

from backend.dataset_conversion.models import CanonicalDataset
from backend.dataset_quality.interfaces import DatasetValidatorInterface


class DatasetValidator(DatasetValidatorInterface):
    async def validate(
        self,
        dataset: CanonicalDataset,
        image_dir: str | None = None,
        ontology_classes: Sequence[str] | None = None,
    ) -> dict[str, object]:
        results: dict[str, object] = {}

        if dataset.image_count > 0:
            results["has_images"] = True
        else:
            results["has_images"] = False
            results["error"] = "Dataset contains no images"

        if dataset.annotation_count > 0:
            results["has_annotations"] = True
        else:
            results["has_annotations"] = False

        if dataset.class_count > 0:
            results["has_classes"] = True
        else:
            results["has_classes"] = False

        annotation_ids = {ann.id for ann in dataset.annotations}
        if len(annotation_ids) < dataset.annotation_count:
            results["duplicate_annotation_ids"] = True
        else:
            results["duplicate_annotation_ids"] = False

        image_ids = {img.id for img in dataset.images}
        if len(image_ids) < dataset.image_count:
            results["duplicate_image_ids"] = True
        else:
            results["duplicate_image_ids"] = False

        invalid_refs: list[str] = []
        valid_image_ids = image_ids
        for ann in dataset.annotations:
            if ann.image_id not in valid_image_ids:
                invalid_refs.append(ann.id)
        results["broken_references"] = invalid_refs

        if not invalid_refs:
            results["all_references_valid"] = True
        else:
            results["all_references_valid"] = False

        results["image_count"] = dataset.image_count
        results["annotation_count"] = dataset.annotation_count
        results["class_count"] = dataset.class_count

        return results
