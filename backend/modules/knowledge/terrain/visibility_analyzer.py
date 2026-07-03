"""Visibility analysis for the Terrain Knowledge pipeline.

Evaluates visibility, observation quality, cover, and concealment
from terrain features and data.  No confidence scoring.
"""

import logging
from typing import Any

from backend.modules.knowledge.terrain.config import terrain_config
from backend.modules.knowledge.terrain.interfaces import (
    TerrainData,
    TerrainFeature,
    VisibilityAnalyzerInterface,
    VisibilityAssessment,
)

logger = logging.getLogger("dss.knowledge.terrain.visibility_analyzer")


class VisibilityAnalyzer(VisibilityAnalyzerInterface):
    """Analyses visibility conditions from terrain features and data.

    Produces a ``VisibilityAssessment`` with visibility rating,
    observation quality, cover, and concealment.
    """

    COVER_FEATURES = {"forest", "building", "urban"}
    CONCEALMENT_FEATURES = {"vegetation", "forest", "obstacle"}

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or terrain_config

    def analyze(
        self,
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> VisibilityAssessment:
        """Assess visibility conditions from features and raw data."""
        visibility = self._assess_visibility(features, data)
        observation = self._assess_observation(features, data)
        cover = self._assess_cover(features)
        concealment = self._assess_concealment(features)
        description = self._build_description(visibility, observation, cover, concealment)

        return VisibilityAssessment(
            visibility=visibility,
            observation=observation,
            cover=cover,
            concealment=concealment,
            description=description,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assess_visibility(
        self,
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> str:
        """Assess overall visibility as ``good``, ``obscured``, or ``blocked``."""
        blocking_features = 0
        for feat in features:
            if feat.feature_type in ("vegetation", "urban", "obstacle", "weather"):
                blocking_features += 1
        for item in data:
            if item.weather and any(
                w in str(item.weather).lower() for w in ("fog", "rain", "snow", "smoke")
            ):
                blocking_features += 2
        if blocking_features >= 3:
            return "blocked"
        if blocking_features >= 1:
            return "obscured"
        return "good"

    @staticmethod
    def _assess_observation(
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> str:
        """Assess observation quality."""
        has_high_ground = False
        for feat in features:
            if any(w in feat.description.lower() for w in ("hill", "ridge", "elevation")):
                has_high_ground = True
                break
        for item in data:
            if item.elevation and item.elevation > 100:
                has_high_ground = True
                break
        return "good" if has_high_ground else "limited"

    @staticmethod
    def _assess_cover(features: list[TerrainFeature]) -> str:
        """Assess available cover from terrain features."""
        cover_sources = [f for f in features if f.feature_type in VisibilityAnalyzer.COVER_FEATURES]
        if len(cover_sources) >= 3:
            return "abundant"
        if cover_sources:
            return "partial"
        return "none"

    @staticmethod
    def _assess_concealment(features: list[TerrainFeature]) -> str:
        """Assess available concealment from terrain features."""
        concealment_sources = [
            f for f in features
            if f.feature_type in VisibilityAnalyzer.CONCEALMENT_FEATURES
        ]
        if len(concealment_sources) >= 3:
            return "abundant"
        if concealment_sources:
            return "partial"
        return "none"

    @staticmethod
    def _build_description(
        visibility: str,
        observation: str,
        cover: str,
        concealment: str,
    ) -> str:
        """Build a human-readable visibility description."""
        return (
            f"Visibility: {visibility}. "
            f"Observation: {observation}. "
            f"Cover: {cover}. "
            f"Concealment: {concealment}."
        )
