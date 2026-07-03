"""Intelligence correlation for the Fusion Engine.

Finds relationships between friendly, enemy, and terrain analyses.
Identifies supporting evidence and builds a unified intelligence
picture.  No confidence scoring or recommendations.
"""

import logging
from typing import Any

from backend.modules.fusion_engine.config import fusion_config
from backend.modules.fusion_engine.interfaces import (
    CollectedIntelligence,
    CorrelatedEvidence,
    CorrelationEngineInterface,
)

logger = logging.getLogger("dss.fusion.correlation_engine")


class CorrelationEngine(CorrelationEngineInterface):
    """Correlates intelligence across all three knowledge domains.

    Finds connections between:
      - Friendly and enemy assessments (e.g. same object, different ID)
      - Terrain conditions and mobility assumptions in analyses
      - Supporting evidence across domains
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or fusion_config

    def correlate(
        self,
        collected: CollectedIntelligence,
    ) -> CorrelatedEvidence:
        """Correlate intelligence across all three domains."""
        correlations: list[str] = []
        supporting_evidence: list[str] = []

        self._correlate_friendly_enemy(collected, correlations, supporting_evidence)
        self._correlate_enemy_terrain(collected, correlations, supporting_evidence)
        self._correlate_friendly_terrain(collected, correlations, supporting_evidence)
        self._extract_domain_evidence(collected, supporting_evidence)

        return CorrelatedEvidence(
            correlations=correlations[:self._config.max_correlations],
            supporting_evidence=supporting_evidence,
        )

    # ------------------------------------------------------------------
    # Private correlators
    # ------------------------------------------------------------------

    @staticmethod
    def _correlate_friendly_enemy(
        collected: CollectedIntelligence,
        correlations: list[str],
        supporting_evidence: list[str],
    ) -> None:
        """Correlate friendly and enemy analyses."""
        f = collected.friendly
        e = collected.enemy

        if f.friendly_match and e.enemy_match:
            correlations.append(
                "Object identified as both friendly and enemy — possible "
                "IFF ambiguity or shared platform type"
            )
            supporting_evidence.append(
                f"Friendly: {f.reason[:120]}"
            )
            supporting_evidence.append(
                f"Enemy: {e.reason[:120]}"
            )
        elif e.enemy_match and not f.friendly_match:
            correlations.append(
                "Enemy match confirmed with no friendly identification"
            )
        elif f.friendly_match and not e.enemy_match:
            correlations.append(
                "Friendly match confirmed with no enemy identification"
            )

    @staticmethod
    def _correlate_enemy_terrain(
        collected: CollectedIntelligence,
        correlations: list[str],
        supporting_evidence: list[str],
    ) -> None:
        """Correlate enemy analysis with terrain conditions."""
        e = collected.enemy
        t = collected.terrain

        if e.enemy_match and e.possible_equipment:
            vehicle = e.possible_equipment
            if not t.road_access and ("tank" in vehicle.lower() or "vehicle" in vehicle.lower()):
                correlations.append(
                    f"Enemy {vehicle} detected in area with no road access — "
                    "mobility may be limited"
                )
                supporting_evidence.append(
                    f"Terrain: road_access={t.road_access}, features={t.nearby_features}"
                )
            if t.visibility in ("obscured", "blocked") and e.enemy_match:
                correlations.append(
                    "Reduced visibility may conceal enemy positions or movements"
                )

    @staticmethod
    def _correlate_friendly_terrain(
        collected: CollectedIntelligence,
        correlations: list[str],
        supporting_evidence: list[str],
    ) -> None:
        """Correlate friendly analysis with terrain conditions."""
        f = collected.friendly
        t = collected.terrain

        if f.friendly_match and t.visibility in ("obscured", "blocked"):
            correlations.append(
                "Friendly forces identified in reduced-visibility terrain — "
                "risk of misidentification increased"
            )
            supporting_evidence.append(
                f"Terrain visibility: {t.visibility}"
            )

    @staticmethod
    def _extract_domain_evidence(
        collected: CollectedIntelligence,
        supporting_evidence: list[str],
    ) -> None:
        """Extract key evidence from each domain analysis."""
        f = collected.friendly
        e = collected.enemy
        t = collected.terrain

        if f.friendly_match:
            supporting_evidence.append(f"Friendly assessment: {f.reason[:120]}")
        if e.enemy_match:
            supporting_evidence.append(f"Enemy assessment: {e.reason[:120]}")
        supporting_evidence.append(
            f"Terrain: {t.terrain_type.value}, visibility={t.visibility}, "
            f"road_access={t.road_access}"
        )
