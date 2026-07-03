"""Evidence construction for enemy-force assessments.

Translates matches between detected objects and known enemy intelligence
into structured, explainable evidence items.  No confidence scoring is
performed.
"""

import logging
from typing import Any

from backend.contracts.models.detection import DetectedObject
from backend.modules.knowledge.enemy.config import enemy_config
from backend.modules.knowledge.enemy.interfaces import (
    Evidence,
    EvidenceBuilderInterface,
    KnowledgeItem,
)

logger = logging.getLogger("dss.knowledge.enemy.evidence_builder")


class EvidenceBuilder(EvidenceBuilderInterface):
    """Builds evidence by comparing a detected object against intelligence items.

    Evidence types produced:

    * ``vehicle_id``         — equipment list matches the detected type.
    * ``platform_match``     — specific platform identification.
    * ``country_attribution`` — country of origin matches detected description.
    * ``weapon_system``      — known weapon system identified.
    * ``camouflage_match``   — camouflage or paint scheme matches.
    * ``capability_match``   — known enemy capability identified.
    * ``threat_indicator``   — known threat indicator observed.
    * ``tactical_role``      — tactical role matches detected context.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or enemy_config

    def build_evidence(
        self,
        obj: DetectedObject,
        knowledge: list[KnowledgeItem],
    ) -> list[Evidence]:
        """Create evidence items linking *obj* to the provided *intelligence*.

        Each intelligence item is compared against the detected object's
        type label and description.  The method is stateless — all
        evidence is derived from the two inputs.
        """
        evidence: list[Evidence] = []
        max_items = self._config.max_evidence_per_object

        for item in knowledge:
            if len(evidence) >= max_items:
                break

            evidence.extend(self._match_vehicle(obj, item))
            evidence.extend(self._match_platform(obj, item))
            evidence.extend(self._match_country(obj, item))
            evidence.extend(self._match_weapon_system(obj, item))
            evidence.extend(self._match_camouflage(obj, item))
            evidence.extend(self._match_capability(obj, item))
            evidence.extend(self._match_threat_indicator(obj, item))
            evidence.extend(self._match_tactical_role(obj, item))

        return evidence

    # ------------------------------------------------------------------
    # Private matchers
    # ------------------------------------------------------------------

    def _match_vehicle(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object ontology type against known enemy equipment."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        for equip in item.equipment:
            equip_lower = equip.lower()
            if equip_lower in obj_type:
                result.append(Evidence(
                    evidence_type="vehicle_id",
                    description=f"Detected '{obj_type}' matches known enemy equipment '{equip}'",
                    matched_attribute=equip,
                    knowledge_source=item.source,
                    weight=0.85,
                ))
        return result

    def _match_platform(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected ontology type against a specific platform identifier."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        if item.unit_name:
            name_lower = item.unit_name.lower()
            if name_lower in obj_type:
                result.append(Evidence(
                    evidence_type="platform_match",
                    description=f"Detected object matches platform '{item.unit_name}'",
                    matched_attribute=item.unit_name,
                    knowledge_source=item.source,
                    weight=0.9,
                ))
        return result

    def _match_country(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object type against known country of origin."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        if item.country and item.country.lower() in obj_type:
            result.append(Evidence(
                evidence_type="country_attribution",
                description=f"Object type '{obj_type}' suggests country of origin '{item.country}'",
                matched_attribute=item.country,
                knowledge_source=item.source,
                weight=0.5,
            ))
        return result

    def _match_weapon_system(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object type against known weapon systems."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        for cap in item.capabilities:
            cap_lower = cap.lower()
            if cap_lower in obj_type:
                result.append(Evidence(
                    evidence_type="weapon_system",
                    description=f"Object type '{obj_type}' matches weapon system '{cap}'",
                    matched_attribute=cap,
                    knowledge_source=item.source,
                    weight=0.6,
                ))
        return result

    def _match_camouflage(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object type against known camouflage or markings.
        Note: CV output contains no color/marking info. This matcher relies
        on object type + geospatial context."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        for marking in item.markings:
            if marking.lower() in obj_type:
                result.append(Evidence(
                    evidence_type="camouflage_match",
                    description=f"Object type '{obj_type}' consistent with marking '{marking}'",
                    matched_attribute=marking,
                    knowledge_source=item.source,
                    weight=0.4,
                ))
        return result

    def _match_capability(self, obj: DetectedObject, item: KnowledgeItem) -> list[Evidence]:
        """Match detected object type against known enemy capabilities."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        for char in item.characteristics:
            if char.lower() in obj_type:
                result.append(Evidence(
                    evidence_type="capability_match",
                    description=f"Object type '{obj_type}' matches characteristic '{char}'",
                    matched_attribute=char,
                    knowledge_source=item.source,
                    weight=0.5,
                ))
        return result

    def _match_threat_indicator(
        self, obj: DetectedObject, item: KnowledgeItem
    ) -> list[Evidence]:
        """Match detected object type against known threat indicators."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        for indicator in item.threat_indicators:
            if indicator.lower() in obj_type:
                result.append(Evidence(
                    evidence_type="threat_indicator",
                    description=f"Object type '{obj_type}' matches threat indicator '{indicator}'",
                    matched_attribute=indicator,
                    knowledge_source=item.source,
                    weight=0.6,
                ))
        return result

    def _match_tactical_role(
        self, obj: DetectedObject, item: KnowledgeItem
    ) -> list[Evidence]:
        """Match detected object type against a known tactical role."""
        result: list[Evidence] = []
        obj_type = obj.object_type.value.lower()

        if item.tactical_role and item.tactical_role.lower() in obj_type:
            result.append(Evidence(
                evidence_type="tactical_role",
                description=f"Ontology type '{obj_type}' matches role '{item.tactical_role}'",
                matched_attribute=item.tactical_role,
                knowledge_source=item.source,
                weight=0.5,
            ))
        return result
