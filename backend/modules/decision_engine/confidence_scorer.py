"""Confidence scoring for the Decision Engine.

Combines fusion confidence, threat confidence, and situation
confidence into a single recommendation confidence score.
Replaceable via ``ConfidenceScorerInterface``.
"""

import logging
from typing import Any

from backend.modules.decision_engine.config import decision_config

logger = logging.getLogger("dss.decision.confidence_scorer")


class ConfidenceScorer:
    """Weighted-average confidence scorer for decision recommendations.

    Combined confidence = (fusion_weight * fusion_confidence)
                        + (threat_weight * threat_confidence)
                        + (situation_weight * situation_confidence)
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or decision_config

    def score(
        self,
        fusion_confidence: float,
        threat_confidence: float,
        situation_confidence: float,
    ) -> float:
        """Calculate a combined confidence score in [0.0, 1.0]."""
        score = (
            self._config.confidence_weight_fusion * fusion_confidence
            + self._config.confidence_weight_threat * threat_confidence
            + self._config.confidence_weight_situation * situation_confidence
        )

        clamped = max(0.0, min(1.0, score))
        logger.debug(
            "Recommendation confidence: fusion=%.2f threat=%.2f "
            "situation=%.2f -> %.2f",
            fusion_confidence,
            threat_confidence,
            situation_confidence,
            clamped,
        )
        return clamped
