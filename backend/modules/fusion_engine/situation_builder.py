"""Situation-building for the Fusion Engine.

Constructs the Common Operational Picture from correlated
and conflict-resolved intelligence.  Produces summaries,
observations, and context.  No recommendations.
"""

import logging

from backend.modules.fusion_engine.interfaces import (
    CollectedIntelligence,
    ConflictRecord,
    CorrelatedEvidence,
    SituationBuilderInterface,
    SituationReport,
)

logger = logging.getLogger("dss.fusion.situation_builder")


class SituationBuilder(SituationBuilderInterface):
    """Builds the Common Operational Picture from fused intelligence.

    Produces:
      - Situation summary (concise one-line assessment)
      - Supporting evidence (list of key evidence items)
      - Key observations (salient findings)
      - Operational context (mission-relevant conditions)
    """

    def build(
        self,
        collected: CollectedIntelligence,
        evidence: CorrelatedEvidence,
        conflicts: ConflictRecord,
    ) -> SituationReport:
        """Build the Common Operational Picture from all intelligence."""
        key_observations = self._build_observations(collected)
        operational_context = self._build_context(collected)
        summary = self._build_summary(collected, evidence, conflicts)

        all_evidence: list[str] = []
        all_evidence.extend(evidence.supporting_evidence)
        all_evidence.extend(conflicts.conflicts)
        all_evidence.extend(conflicts.resolutions)

        return SituationReport(
            summary=summary,
            supporting_evidence=all_evidence,
            key_observations=key_observations,
            operational_context=operational_context,
        )

    # ------------------------------------------------------------------
    # Private builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        collected: CollectedIntelligence,
        evidence: CorrelatedEvidence,
        conflicts: ConflictRecord,
    ) -> str:
        """Build a concise situation summary."""
        f = collected.friendly
        e = collected.enemy
        t = collected.terrain

        parts: list[str] = []

        if e.enemy_match:
            parts.append(f"Enemy presence detected (confidence={e.confidence:.2f})")
            if e.possible_equipment:
                parts.append(f"equipment: {e.possible_equipment}")
        elif f.friendly_match:
            parts.append(f"Friendly presence confirmed (confidence={f.confidence:.2f})")
        else:
            parts.append("No immediate threat identified")

        parts.append(f"Terrain: {t.terrain_type.value}")
        if t.road_access:
            parts.append("road access available")
        else:
            parts.append("no road access")

        if conflicts.conflicts:
            parts.append(f"{len(conflicts.conflicts)} conflict(s) resolved")

        if evidence.correlations:
            parts.append(f"{len(evidence.correlations)} correlation(s) found")

        return " | ".join(parts)

    @staticmethod
    def _build_observations(
        collected: CollectedIntelligence,
    ) -> list[str]:
        """Build key observations from the intelligence."""
        observations: list[str] = []
        f = collected.friendly
        e = collected.enemy
        t = collected.terrain

        if e.enemy_match:
            observations.append(f"Enemy force identified: {e.reason[:100]}")
        if f.friendly_match:
            observations.append(f"Friendly force identified: {f.reason[:100]}")
        observations.append(
            f"Terrain classified as {t.terrain_type.value} "
            f"with visibility '{t.visibility}'"
        )
        if t.elevation is not None:
            observations.append(f"Area elevation: {t.elevation}m")

        return observations

    @staticmethod
    def _build_context(
        collected: CollectedIntelligence,
    ) -> str:
        """Build operational context from the intelligence."""
        f = collected.friendly
        e = collected.enemy
        t = collected.terrain

        context_parts: list[str] = []

        if e.enemy_match and f.friendly_match:
            context_parts.append("CONTACT SITUATION — friendly and enemy forces co-located")
        elif e.enemy_match:
            context_parts.append("HOSTILE SITUATION — enemy forces present")
        elif f.friendly_match:
            context_parts.append("FRIENDLY SITUATION — only friendly forces identified")

        context_parts.append(f"Terrain: {t.terrain_type.value}")
        context_parts.append(f"Visibility: {t.visibility}")
        context_parts.append(f"Road access: {'yes' if t.road_access else 'no'}")

        if t.nearby_features:
            context_parts.append(f"Features: {', '.join(t.nearby_features)}")

        return " | ".join(context_parts)
