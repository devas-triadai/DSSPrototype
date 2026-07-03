"""Confidence scoring for terrain assessments.

Produces a final confidence score by combining detection confidence,
terrain-source confidence, and feature evidence strength.  The
weighting formula is fully replaceable via ``ConfidenceScorerInterface``.
"""

import logging
from typing import Any

from backend.modules.knowledge.terrain.config import terrain_config

logger = logging.getLogger("dss.knowledge.terrain.confidence_scorer")


class ConfidenceScorer:
    """Weighted-average confidence scorer.

    Final confidence = (detection_weight * detection_confidence)
                     + (terrain_weight * terrain_confidence)
                     + (evidence_weight * evidence_confidence)

    Where *evidence_confidence* is derived from the number and quality
    of extracted terrain features.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or terrain_config

    def score(
        self,
        detection_confidence: float,
        terrain_confidence: float,
        evidence_confidence: float,
    ) -> float:
        """Calculate a weighted confidence score in [0.0, 1.0]."""
        score = (
            self._config.confidence_weight_detection * detection_confidence
            + self._config.confidence_weight_terrain * terrain_confidence
            + self._config.confidence_weight_evidence * evidence_confidence
        )

        clamped = max(0.0, min(1.0, score))
        logger.debug(
            "Confidence: detection=%.2f terrain=%.2f evidence=%.2f -> %.2f",
            detection_confidence,
            terrain_confidence,
            evidence_confidence,
            clamped,
        )
        return clamped
