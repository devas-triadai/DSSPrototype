"""Intelligence validation for the Fusion Engine.

Validates collected intelligence for completeness, correctness,
and structural integrity before correlation.
"""

import logging
from typing import Any

from backend.modules.fusion_engine.config import fusion_config
from backend.modules.fusion_engine.interfaces import (
    CollectedIntelligence,
    ValidationResult,
    ValidatorInterface,
)

logger = logging.getLogger("dss.fusion.validator")


class Validator(ValidatorInterface):
    """Validates collected intelligence from all three domain modules.

    Checks performed:
      - All three analysis objects are present.
      - Required string fields are non-empty.
      - Confidence values are in [0, 1].
      - Boolean fields are set.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or fusion_config

    def validate(
        self,
        collected: CollectedIntelligence,
    ) -> ValidationResult:
        """Validate collected intelligence for completeness and correctness."""
        issues: list[str] = []

        self._validate_friendly(collected.friendly, issues)
        self._validate_enemy(collected.enemy, issues)
        self._validate_terrain(collected.terrain, issues)

        valid = len(issues) == 0
        if not valid:
            for issue in issues:
                logger.warning("Validation issue: %s", issue)

        return ValidationResult(valid=valid, issues=issues)

    # ------------------------------------------------------------------
    # Private validators
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_friendly(friendly: Any, issues: list[str]) -> None:
        if friendly is None:
            issues.append("FriendlyAnalysis is missing")
            return
        if not friendly.reason:
            issues.append("FriendlyAnalysis.reason is empty")
        if not (0.0 <= friendly.confidence <= 1.0):
            issues.append(f"FriendlyAnalysis.confidence out of range: {friendly.confidence}")
        if not isinstance(friendly.friendly_match, bool):
            issues.append("FriendlyAnalysis.friendly_match must be a boolean")

    @staticmethod
    def _validate_enemy(enemy: Any, issues: list[str]) -> None:
        if enemy is None:
            issues.append("EnemyAnalysis is missing")
            return
        if not enemy.reason:
            issues.append("EnemyAnalysis.reason is empty")
        if not (0.0 <= enemy.confidence <= 1.0):
            issues.append(f"EnemyAnalysis.confidence out of range: {enemy.confidence}")
        if not isinstance(enemy.enemy_match, bool):
            issues.append("EnemyAnalysis.enemy_match must be a boolean")

    @staticmethod
    def _validate_terrain(terrain: Any, issues: list[str]) -> None:
        if terrain is None:
            issues.append("TerrainAnalysis is missing")
            return
        if not terrain.reason:
            issues.append("TerrainAnalysis.reason is empty")
        if not terrain.visibility:
            issues.append("TerrainAnalysis.visibility is empty")
        if not isinstance(terrain.road_access, bool):
            issues.append("TerrainAnalysis.road_access must be a boolean")
        if terrain.terrain_type is None:
            issues.append("TerrainAnalysis.terrain_type is missing")
