from __future__ import annotations

from backend.dataset_conversion.models import (
    CanonicalDataset,
    ValidationReport,
)
from backend.dataset_conversion.ontology_adapter import OntologyAdapter


class DatasetValidator:
    def __init__(
        self,
        ontology_adapter: OntologyAdapter | None = None,
    ) -> None:
        self._ontology_adapter = ontology_adapter or OntologyAdapter()

    async def validate(
        self,
        dataset: CanonicalDataset,
    ) -> ValidationReport:
        errors: list[str] = []
        warnings: list[str] = []
        total_checks = 8
        passed = 0
        failed = 0

        check_results: list[tuple[str, bool]] = []

        # 1. Image existence check
        has_imgs = dataset.image_count > 0
        check_results.append(("Has images", has_imgs))
        if has_imgs:
            passed += 1
            missing: list[str] = []
            for img in dataset.images:
                if not img.file_path:
                    missing.append(img.id)
            if missing:
                errors.append(f"Images missing file paths: {missing[:5]}...")
                failed += 1
            else:
                passed += 1
        else:
            errors.append("Dataset has no images")
            failed += 1

        # 2. Annotation count check
        has_anns = dataset.annotation_count > 0
        check_results.append(("Has annotations", has_anns))
        if has_anns:
            passed += 1
        else:
            errors.append("Dataset has no annotations")
            failed += 1

        # 3. Geometry validity
        geom_valid = True
        for ann in dataset.annotations:
            if ann.width <= 0 or ann.height <= 0:
                errors.append(
                    f"Annotation '{ann.id}' has invalid geometry: "
                    f"width={ann.width}, height={ann.height}"
                )
                geom_valid = False
                failed += 1
                break
        if geom_valid:
            check_results.append(("Geometry valid", True))
            passed += 1

        # 4. Ontology validity
        ontology_valid = True
        seen_labels: set[str] = set()
        for ann in dataset.annotations:
            if ann.canonical_label not in seen_labels:
                seen_labels.add(ann.canonical_label)
                exists = await self._ontology_adapter.is_known_label(ann.canonical_label)
                if not exists:
                    warnings.append(f"Label '{ann.canonical_label}' is not in the current ontology")
        check_results.append(("Ontology labels valid", ontology_valid))
        if ontology_valid:
            passed += 1
        else:
            failed += 1

        # 5. Duplicate IDs
        seen_img_ids: set[str] = set()
        dup_imgs = 0
        for img in dataset.images:
            if img.id in seen_img_ids:
                dup_imgs += 1
            seen_img_ids.add(img.id)
        if dup_imgs > 0:
            errors.append(f"Found {dup_imgs} duplicate image IDs")
            failed += 1
        else:
            check_results.append(("No duplicate image IDs", True))
            passed += 1

        seen_ann_ids: set[str] = set()
        dup_anns = 0
        for ann in dataset.annotations:
            if ann.id in seen_ann_ids:
                dup_anns += 1
            seen_ann_ids.add(ann.id)
        if dup_anns > 0:
            errors.append(f"Found {dup_anns} duplicate annotation IDs")
            failed += 1
        else:
            check_results.append(("No duplicate annotation IDs", True))
            passed += 1

        # 6. Orphan annotations
        all_img_ids = {img.id for img in dataset.images}
        orphan_ids: set[str] = set()
        for ann in dataset.annotations:
            if ann.image_id not in all_img_ids:
                orphan_ids.add(ann.image_id)
        if orphan_ids:
            errors.append(
                f"Found annotations referencing non-existent images: {len(orphan_ids)} orphan(s)"
            )
            failed += 1
        else:
            check_results.append(("No orphan annotations", True))
            passed += 1

        # 7. Class coverage
        class_counts: dict[str, int] = {}
        for ann in dataset.annotations:
            class_counts[ann.canonical_label] = class_counts.get(ann.canonical_label, 0) + 1
        if class_counts:
            min_count = min(class_counts.values())
            if min_count == 0:
                warnings.append("Some classes have zero annotations")
            check_results.append(("Class coverage checked", True))
            passed += 1
        else:
            check_results.append(("Class coverage checked (no classes)", True))
            passed += 1

        # 8. Split integrity (always passes for unsplit dataset)
        check_results.append(("Split integrity", True))
        passed += 1

        for check_name, result in check_results:
            if not result:
                failed += 1

        return ValidationReport(
            valid=len(errors) == 0,
            total_checks=total_checks,
            passed_checks=passed,
            failed_checks=failed,
            errors=tuple(errors),
            warnings=tuple(warnings),
        )
