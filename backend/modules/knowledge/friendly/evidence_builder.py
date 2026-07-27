"""Evidence construction for friendly-force assessments.

Translates matches between detected objects and known friendly
knowledge into structured, explainable evidence items.  No confidence
scoring is performed.
"""

import logging
from typing import Any

from backend.contracts.models.detection import DetectedObject
from backend.modules.knowledge.friendly.config import friendly_config
from backend.modules.knowledge.friendly.interfaces import (
    Evidence,
    EvidenceBuilderInterface,
    KnowledgeItem,
)

logger = logging.getLogger("dss.knowledge.friendly.evidence_builder")


class EvidenceBuilder(EvidenceBuilderInterface):
    """Builds evidence by comparing a detected object against knowledge items.

    Evidence types produced:

    * ``vehicle_match``    — equipment list matches the detected type.
    * ``marking_match``    — markings on the object match known friendly markings.
    * ``characteristic_match`` — physical characteristics match.
    * ``identifier_match`` — unit ID or name matches.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or friendly_config

    def build_evidence(
        self,
        obj: DetectedObject,
        knowledge: list[KnowledgeItem],
    ) -> list[Evidence]:
        """Create evidence items linking *obj* to the provided *knowledge*.

        Each knowledge item is compared against the detected object's
        type label and description.  The method is stateless — all
        evidence is derived from the two inputs.
        """
        evidence: list[Evidence] = []
        max_items = self._config.max_evidence_per_object

        for item in knowledge:
            if len(evidence) >= max_items:
                break

            evidence.extend(self._match_equipment(obj, item))
            evidence.extend(self._match_markings(obj, item))
            evidence.extend(self._match_characteristics(obj, item))
            evidence.extend(self._match_identifiers(obj, item))

        return evidence

    # ------------------------------------------------------------------
    # Private matchers
    # ------------------------------------------------------------------

    def _match_equipment(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object ontology type against known equipment."""
        result: list[Evidence] = []
        obj_type = obj.name.lower()

        for equip in item.equipment:
            equip_lower = equip.lower()
            if equip_lower in obj_type:
                result.append(Evidence(
                    evidence_type="vehicle_match",
                    description=f"Detected '{obj_type}' matches known equipment '{equip}'",
                    matched_attribute=equip,
                    knowledge_source=item.source,
                    weight=0.8,
                ))
        return result

    def _match_markings(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object type against known markings.
        Note: Marking/color info requires higher-resolution imagery.
        Fallback uses object type + geospatial proximity.
        """
        result: list[Evidence] = []
        obj_type = obj.name.lower()

        for marking in item.markings:
            if marking.lower() in obj_type:
                result.append(Evidence(
                    evidence_type="marking_match",
                    description=f"Object type '{obj_type}' consistent with marking '{marking}'",
                    matched_attribute=marking,
                    knowledge_source=item.source,
                    weight=0.4,
                ))
        return result

    def _match_characteristics(
        self, obj: DetectedObject, item: KnowledgeItem
    ) -> list[Evidence]:
        """Match detected object type against known characteristics."""
        result: list[Evidence] = []
        obj_type = obj.name.lower()

        for char in item.characteristics:
            if char.lower() in obj_type:
                result.append(Evidence(
                    evidence_type="characteristic_match",
                    description=f"Object type '{obj_type}' matches characteristic '{char}'",
                    matched_attribute=char,
                    knowledge_source=item.source,
                    weight=0.5,
                ))
        return result

    def _match_identifiers(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object type against unit identifiers."""
        result: list[Evidence] = []
        obj_type = obj.name.lower()

        identifiers: list[str] = []
        if item.unit_id:
            identifiers.append(item.unit_id)
        if item.unit_name:
            identifiers.append(item.unit_name)

        for ident in identifiers:
            if ident.lower() in obj_type:
                result.append(Evidence(
                    evidence_type="identifier_match",
                    description=f"Description contains unit identifier '{ident}'",
                    matched_attribute=ident,
                    knowledge_source=item.source,
                    weight=0.9,
                ))
        return result
