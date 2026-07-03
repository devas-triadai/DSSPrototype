"""Situation evaluation for the Decision Engine.

Evaluates the operational situation from fused intelligence and
threat assessment.  Produces structured context.  No recommendations.
"""

import logging
from typing import Any

from backend.contracts.enums.core import ThreatLevel
from backend.contracts.models.fusion import FusionResult, ThreatAssessment
from backend.modules.decision_engine.config import decision_config
from backend.modules.decision_engine.interfaces import (
    SituationContext,
    SituationEvaluatorInterface,
)

logger = logging.getLogger("dss.decision.situation_evaluator")


class SituationEvaluator(SituationEvaluatorInterface):
    """Evaluates the operational situation from fused intelligence.

    Produces a ``SituationContext`` with:
      - Threat level and confidences
      - Enemy / friendly presence flags
      - Terrain summary extracted from fusion summary
      - Key observations from supporting evidence
      - Severity rating
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or decision_config

    def evaluate(
        self,
        fusion: FusionResult,
        threat: ThreatAssessment,
    ) -> SituationContext:
        """Evaluate the operational situation from fused intelligence."""
        has_enemy = self._detect_enemy(fusion)
        has_friendly = self._detect_friendly(fusion)
        terrain_summary = self._extract_terrain(fusion)
        severity = self._assess_severity(threat, has_enemy, has_friendly)
        observations = self._build_observations(fusion, threat, has_enemy, has_friendly)

        return SituationContext(
            threat_level=threat.threat_level,
            threat_confidence=threat.confidence,
            fusion_confidence=fusion.combined_confidence,
            has_enemy=has_enemy,
            has_friendly=has_friendly,
            terrain_summary=terrain_summary,
            key_observations=observations,
            severity=severity,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_enemy(fusion: FusionResult) -> bool:
        """Detect enemy presence from the fusion summary."""
        return "enemy" in fusion.summary.lower()

    @staticmethod
    def _detect_friendly(fusion: FusionResult) -> bool:
        """Detect friendly presence from the fusion summary."""
        return "friendly" in fusion.summary.lower()

    @staticmethod
    def _extract_terrain(fusion: FusionResult) -> str:
        """Extract terrain information from the fusion summary."""
        summary = fusion.summary.lower()
        terrain_keywords = [
            "terrain", "open_field", "forest", "urban", "river",
            "hill", "road", "bridge", "visibility", "elevation",
        ]
        terrain_parts = [
            part.strip() for part in summary.split("|")
            if any(kw in part for kw in terrain_keywords)
        ]
        return "; ".join(terrain_parts) if terrain_parts else "unknown"

    def _assess_severity(
        self,
        threat: ThreatAssessment,
        has_enemy: bool,
        has_friendly: bool,
    ) -> str:
        """Assess situation severity from threat and presence of forces."""
        if not has_enemy and not has_friendly:
            return "low"
        if not has_enemy:
            return "low"

        base = self._threat_level_score(threat.threat_level)
        confidence_modifier = threat.confidence - 0.5
        friendly_modifier = -0.2 if has_friendly else 0.0
        score = base + confidence_modifier + friendly_modifier

        if score >= self._config.severity_high_threshold:
            return "critical"
        if score >= self._config.severity_medium_threshold:
            return "high"
        return "medium"

    @staticmethod
    def _threat_level_score(level: ThreatLevel) -> float:
        """Map ThreatLevel to a numeric score."""
        mapping = {
            ThreatLevel.CRITICAL: 1.0,
            ThreatLevel.HIGH: 0.8,
            ThreatLevel.MEDIUM: 0.5,
            ThreatLevel.LOW: 0.2,
            ThreatLevel.UNKNOWN: 0.3,
        }
        return mapping.get(level, 0.3)

    @staticmethod
    def _build_observations(
        fusion: FusionResult,
        threat: ThreatAssessment,
        has_enemy: bool,
        has_friendly: bool,
    ) -> list[str]:
        """Build key observations from intelligence."""
        observations: list[str] = []
        observations.append(f"Threat level: {threat.threat_level.value}")
        observations.append(f"Fusion confidence: {fusion.combined_confidence:.2f}")
        if has_enemy:
            observations.append("Enemy presence detected")
        if has_friendly:
            observations.append("Friendly presence detected")
        if fusion.supporting_evidence:
            observations.append(f"Evidence: {len(fusion.supporting_evidence)} item(s)")
        return observations
