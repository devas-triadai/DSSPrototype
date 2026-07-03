"""Class harmonizer — unifies class names across datasets via ontology.

Example: ``tank``, ``MBT``, ``battle tank`` → ``main_battle_tank``

Relies on the ``OntologyMappingReport`` produced by the ontology mapper.
No hardcoded rules; all mappings are derived from the ontology service.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from backend.dataset_intelligence.interfaces import ClassHarmonizerInterface
from backend.dataset_intelligence.models import (
    HarmonizedDataset,
    ImageRecord,
    NormalizedDataset,
    OntologyMappingReport,
)

logger = logging.getLogger("dss.dataset_intelligence.class_harmonizer")


class ClassHarmonizer(ClassHarmonizerInterface):
    """Harmonize class names using ontology mappings."""

    def harmonize(
        self,
        dataset: NormalizedDataset,
        ontology_mapping: OntologyMappingReport,
    ) -> HarmonizedDataset:
        logger.info(
            "Class harmonization started | dataset=%s | mappings=%d",
            dataset.dataset_id,
            len(ontology_mapping.mappings),
        )
        harmonization_mapping = self.build_harmonization_mapping(dataset.classes, ontology_mapping)
        harmonized_classes: set[str] = set()
        harmonized_images: list[ImageRecord] = []

        for img in dataset.images:
            updated_anns = []
            for ann in img.annotations:
                norm_class = ann.normalized_class or ann.class_name
                harmonized = harmonization_mapping.get(norm_class, norm_class)
                harmonized_classes.add(harmonized)
                updated_anns.append(
                    ann.model_copy(
                        update={
                            "normalized_class": norm_class,
                            "ontology_class": harmonized,
                        }
                    )
                )
            updated_provenance = None
            if img.provenance is not None:
                updated_provenance = img.provenance.model_copy(
                    update={
                        "normalized_class": img.provenance.normalized_class
                        or img.provenance.original_class,
                        "ontology_class": harmonization_mapping.get(
                            img.provenance.normalized_class or img.provenance.original_class,
                            img.provenance.normalized_class or img.provenance.original_class,
                        ),
                    }
                )
            harmonized_images.append(
                img.model_copy(
                    update={"annotations": updated_anns, "provenance": updated_provenance}
                )
            )

        result = HarmonizedDataset(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.dataset_name,
            images=harmonized_images,
            classes=sorted(harmonized_classes),
            harmonization_mapping=harmonization_mapping,
            ontology_version=ontology_mapping.ontology_version,
        )
        logger.info(
            "Class harmonization complete | dataset=%s | harmonized_classes=%d",
            dataset.dataset_id,
            len(result.classes),
        )
        return result

    def build_harmonization_mapping(
        self,
        classes: Sequence[str],
        ontology_mapping: OntologyMappingReport,
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for cls in classes:
            mapped = ontology_mapping.mappings.get(cls, cls)
            mapping[cls] = mapped
        return mapping
