"""Confidence scoring for the Fusion Engine.

Combines confidence from all three knowledge domains and cross-domain
correlation into a single score.  The weighting formula is fully
replaceable via ``ConfidenceScorerInterface``.
"""

import logging
from typing import Any

from backend.modules.fusion_engine.config import fusion_config

logger = logging.getLogger("dss.fusion.confidence_scorer")


class ConfidenceScorer:
    """Weighted-average confidence scorer for fused intelligence.

    Combined confidence = (friendly_weight * friendly_confidence)
                        + (enemy_weight * enemy_confidence)
                        + (terrain_weight * terrain_confidence)
                        + (correlation_weight * correlation_confidence)
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or fusion_config

    def score(
        self,
        friendly_confidence: float,
        enemy_confidence: float,
        terrain_confidence: float,
        correlation_confidence: float,
    ) -> float:
        """Calculate a combined confidence score in [0.0, 1.0]."""
        score = (
            self._config.confidence_weight_friendly * friendly_confidence
            + self._config.confidence_weight_enemy * enemy_confidence
            + self._config.confidence_weight_terrain * terrain_confidence
            + self._config.confidence_weight_correlation * correlation_confidence
        )

        clamped = max(0.0, min(1.0, score))
        logger.debug(
            "Combined confidence: friendly=%.2f enemy=%.2f terrain=%.2f "
            "correlation=%.2f -> %.2f",
            friendly_confidence,
            enemy_confidence,
            terrain_confidence,
            correlation_confidence,
            clamped,
        )
        return clamped
