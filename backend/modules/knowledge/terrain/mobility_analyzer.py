"""Terrain mobility analysis for the Terrain Knowledge pipeline.

Evaluates road access, terrain difficulty, obstacles, and movement
constraints from terrain features and data.  No confidence scoring.
"""

import logging
from typing import Any

from backend.modules.knowledge.terrain.config import terrain_config
from backend.modules.knowledge.terrain.interfaces import (
    MobilityAnalyzerInterface,
    MobilityAssessment,
    TerrainData,
    TerrainFeature,
)

logger = logging.getLogger("dss.knowledge.terrain.mobility_analyzer")


class MobilityAnalyzer(MobilityAnalyzerInterface):
    """Analyses mobility conditions from terrain features and data.

    Produces a ``MobilityAssessment`` with road access, mobility
    rating, terrain difficulty, and identified obstacles.
    """

    OBSTACLE_TYPES = {"obstacle", "water", "bridge"}

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or terrain_config

    def analyze(
        self,
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> MobilityAssessment:
        """Assess mobility conditions from features and raw data."""
        has_road = self._detect_road_access(features, data)
        obstacles = self._collect_obstacles(features)
        rating = self._rate_mobility(features, data, obstacles)
        difficulty = self._assess_difficulty(rating)
        description = self._build_description(has_road, rating, obstacles)

        return MobilityAssessment(
            road_access=has_road,
            mobility_rating=rating,
            terrain_difficulty=difficulty,
            obstacles=obstacles,
            description=description,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_road_access(
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> bool:
        """Return ``True`` if roads are present in features or data."""
        for feat in features:
            if feat.feature_type == "road":
                return True
        for item in data:
            if item.road_network:
                return True
        return False

    def _collect_obstacles(self, features: list[TerrainFeature]) -> list[str]:
        """Collect obstacle descriptions from features."""
        obstacles: list[str] = []
        for feat in features:
            if feat.feature_type in self.OBSTACLE_TYPES:
                obstacles.append(feat.description)
        return obstacles

    def _rate_mobility(
        self,
        features: list[TerrainFeature],
        data: list[TerrainData],
        obstacles: list[str],
    ) -> str:
        """Rate mobility as ``good``, ``limited``, or ``poor``."""
        obstacle_penalty = len(obstacles) * 0.15
        road_bonus = 0.3 if self._detect_road_access(features, data) else 0.0
        terrain_penalty = self._compute_terrain_penalty(features)

        score = road_bonus - obstacle_penalty - terrain_penalty

        if score >= self._config.mobility_rating_high_threshold:
            return "good"
        if score >= self._config.mobility_rating_medium_threshold:
            return "limited"
        return "poor"

    @staticmethod
    def _compute_terrain_penalty(features: list[TerrainFeature]) -> float:
        """Compute a terrain difficulty penalty from features."""
        penalty = 0.0
        for feat in features:
            if feat.feature_type in ("vegetation", "water", "soil"):
                penalty += 0.1
        return min(penalty, 0.8)

    @staticmethod
    def _assess_difficulty(rating: str) -> str:
        """Map mobility rating to a difficulty label."""
        mapping = {
            "good": "easy",
            "limited": "moderate",
            "poor": "difficult",
        }
        return mapping.get(rating, "unknown")

    @staticmethod
    def _build_description(
        has_road: bool,
        rating: str,
        obstacles: list[str],
    ) -> str:
        """Build a human-readable mobility description."""
        parts = [f"Mobility rating: {rating}"]
        parts.append("Road access available." if has_road else "No road access.")
        if obstacles:
            parts.append(f"Obstacles: {', '.join(obstacles)}.")
        return " ".join(parts)
