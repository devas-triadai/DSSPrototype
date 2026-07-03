"""Ontology mapper integrating with the DSS Knowledge Ontology engine.

Maps every normalized class name to its canonical ontology concept via
alias resolution. No hardcoded mappings — all resolution is delegated to
the injected ``OntologyServiceInterface``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from backend.dataset_intelligence.interfaces import OntologyMapperInterface
from backend.dataset_intelligence.models import (
    NormalizedDataset,
    OntologyMappingReport,
)

if TYPE_CHECKING:
    from backend.modules.knowledge.ontology.interfaces import OntologyServiceInterface

logger = logging.getLogger("dss.dataset_intelligence.ontology_mapper")


class OntologyMapper(OntologyMapperInterface):
    """Map dataset classes to canonical ontology concepts.

    Parameters
    ----------
    ontology_service:
        Injectable ontology service. If ``None``, a default
        ``OntologyService`` is constructed on first use.
    """

    def __init__(self, ontology_service: "OntologyServiceInterface | None" = None) -> None:
        self._ontology_service = ontology_service

    def _get_service(self) -> "OntologyServiceInterface":
        if self._ontology_service is None:
            from backend.modules.knowledge.ontology.service import OntologyService

            self._ontology_service = OntologyService()
        return self._ontology_service

    def map_classes(self, dataset: NormalizedDataset) -> OntologyMappingReport:
        logger.info(
            "Ontology mapping started | dataset=%s | classes=%s",
            dataset.dataset_id,
            dataset.classes,
        )
        service = self._get_service()
        mappings: dict[str, str] = {}
        unmapped: list[str] = []

        for cls in dataset.classes:
            try:
                result = service.process(cls, detector_confidence=1.0)
                canonical = result.canonical_concept
                if canonical:
                    mappings[cls] = canonical
                    logger.debug("Ontology map: '%s' -> '%s'", cls, canonical)
                else:
                    unmapped.append(cls)
                    logger.debug("Unmapped class: '%s'", cls)
            except Exception as exc:
                logger.warning("Ontology mapping failed for '%s': %s", cls, exc)
                unmapped.append(cls)

        coverage = len(mappings) / len(dataset.classes) if dataset.classes else 1.0
        report = OntologyMappingReport(
            dataset_id=dataset.dataset_id,
            mappings=mappings,
            unmapped_classes=unmapped,
            ontology_coverage=coverage,
            ontology_version=service.get_version(),
        )
        logger.info(
            "Ontology mapping complete | dataset=%s | mapped=%d | unmapped=%d | coverage=%.2f",
            dataset.dataset_id,
            len(mappings),
            len(unmapped),
            coverage,
        )
        return report

    def apply_mapping(
        self,
        dataset: NormalizedDataset,
        report: OntologyMappingReport,
    ) -> NormalizedDataset:
        """Apply ontology mappings to all annotations in *dataset*.

        Returns a new ``NormalizedDataset`` with updated ``ontology_class``
        fields on every ``Annotation`` and ``ProvenanceRecord``.
        """
        logger.info("Applying ontology mapping | dataset=%s", dataset.dataset_id)
        updated_images = []
        for img in dataset.images:
            updated_anns = []
            for ann in img.annotations:
                ontology_class = report.mappings.get(
                    ann.normalized_class or ann.class_name,
                    ann.normalized_class or ann.class_name,
                )
                updated_anns.append(ann.model_copy(update={"ontology_class": ontology_class}))
            updated_provenance = None
            if img.provenance is not None:
                updated_provenance = img.provenance.model_copy(
                    update={
                        "ontology_class": report.mappings.get(
                            img.provenance.normalized_class or img.provenance.original_class,
                            img.provenance.normalized_class or img.provenance.original_class,
                        )
                    }
                )
            updated_images.append(
                img.model_copy(
                    update={"annotations": updated_anns, "provenance": updated_provenance}
                )
            )

        return NormalizedDataset(
            dataset_id=dataset.dataset_id,
            dataset_name=dataset.dataset_name,
            images=updated_images,
            classes=dataset.classes,
            class_mapping=dataset.class_mapping,
            metadata=dataset.metadata,
            normalization_log=dataset.normalization_log,
        )
