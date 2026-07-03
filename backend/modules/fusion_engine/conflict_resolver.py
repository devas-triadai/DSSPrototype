"""Conflict detection and resolution for the Fusion Engine.

Identifies conflicting intelligence between domain modules and
produces resolved assessments.  No recommendations or confidence
scoring.
"""

import logging
from typing import Any

from backend.modules.fusion_engine.config import fusion_config
from backend.modules.fusion_engine.interfaces import (
    CollectedIntelligence,
    ConflictRecord,
    ConflictResolverInterface,
    CorrelatedEvidence,
)

logger = logging.getLogger("dss.fusion.conflict_resolver")


class ConflictResolver(ConflictResolverInterface):
    """Detects and resolves conflicts between intelligence sources.

    Conflict types detected:
      - Friendly vs enemy identification of the same subject.
      - Terrain conditions contradicting mobility assumptions.
      - Confidence disagreement between sources.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or fusion_config

    def resolve(
        self,
        collected: CollectedIntelligence,
        evidence: CorrelatedEvidence,
    ) -> ConflictRecord:
        """Detect and resolve conflicts between intelligence sources."""
        conflicts: list[str] = []
        resolutions: list[str] = []

        self._detect_identification_conflicts(collected, conflicts, resolutions)
        self._detect_terrain_conflicts(collected, conflicts, resolutions)
        self._detect_confidence_conflicts(collected, conflicts, resolutions)

        return ConflictRecord(conflicts=conflicts, resolutions=resolutions)

    # ------------------------------------------------------------------
    # Private conflict detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_identification_conflicts(
        collected: CollectedIntelligence,
        conflicts: list[str],
        resolutions: list[str],
    ) -> None:
        """Detect conflicts where friendly and enemy both claim the same subject."""
        f = collected.friendly
        e = collected.enemy

        if f.friendly_match and e.enemy_match:
            conflicts.append(
                "Identification conflict: object identified as both friendly and enemy"
            )
            if f.confidence >= e.confidence:
                resolutions.append(
                    f"Resolved by higher friendly confidence "
                    f"({f.confidence:.2f} vs {e.confidence:.2f}) — "
                    "subject treated as friendly with caution"
                )
            else:
                resolutions.append(
                    f"Resolved by higher enemy confidence "
                    f"({e.confidence:.2f} vs {f.confidence:.2f}) — "
                    "subject treated as hostile with caution"
                )

    @staticmethod
    def _detect_terrain_conflicts(
        collected: CollectedIntelligence,
        conflicts: list[str],
        resolutions: list[str],
    ) -> None:
        """Detect conflicts between terrain analysis and domain assessments."""
        e = collected.enemy
        t = collected.terrain

        if e.enemy_match and e.possible_equipment:
            vehicle = e.possible_equipment.lower()
            if not t.road_access and ("tank" in vehicle or "vehicle" in vehicle):
                conflicts.append(
                    f"Terrain conflict: enemy {e.possible_equipment} detected "
                    f"but terrain reports no road access"
                )
                resolutions.append(
                    "Terrain assessment likely correct — "
                    "enemy vehicle may have reached location via off-road "
                    "movement or alternate route"
                )

    @staticmethod
    def _detect_confidence_conflicts(
        collected: CollectedIntelligence,
        conflicts: list[str],
        resolutions: list[str],
    ) -> None:
        """Detect significant disagreements in confidence levels."""
        f = collected.friendly
        e = collected.enemy

        # TerrainAnalysis does not carry a confidence field, so use a
        # proxy derived from terrain data completeness.
        terrain_proxy = fusion_config.default_confidence
        if collected.terrain.terrain_type and collected.terrain.visibility:
            terrain_proxy = min(fusion_config.default_confidence + 0.2, 1.0)

        confidences = [
            ("friendly", f.confidence),
            ("enemy", e.confidence),
            ("terrain", terrain_proxy),
        ]

        for name_a, conf_a in confidences:
            for name_b, conf_b in confidences:
                if name_a < name_b and abs(conf_a - conf_b) > 0.5:
                    conflicts.append(
                        f"Confidence conflict: {name_a} ({conf_a:.2f}) vs "
                        f"{name_b} ({conf_b:.2f})"
                    )
                    resolutions.append(
                        f"Confidence disparity noted — {name_a}={conf_a:.2f}, "
                        f"{name_b}={conf_b:.2f}. Treating with caution."
                    )
