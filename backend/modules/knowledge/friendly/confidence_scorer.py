"""Confidence scoring for friendly-force assessments.

Produces a final confidence score by combining detection confidence,
knowledge-source confidence, and evidence strength.  The weighting
formula is fully replaceable via ``ConfidenceScorerInterface``.
"""

import logging
from typing import Any

from backend.modules.knowledge.friendly.config import friendly_config
from backend.modules.knowledge.friendly.exceptions import ScoringError
from backend.modules.knowledge.friendly.interfaces import (
    ConfidenceScorerInterface,
    Evidence,
)

logger = logging.getLogger("dss.knowledge.friendly.confidence_scorer")


class ConfidenceScorer(ConfidenceScorerInterface):
    """Weighted-average confidence scorer.

    Final confidence = (detection_weight * detection_confidence)
                     + (knowledge_weight * knowledge_confidence)
                     + (evidence_weight * evidence_strength)

    Where *evidence_strength* is the mean weight of all evidence items
    (zero if no evidence).
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or friendly_config

    def score(
        self,
        evidence: list[Evidence],
        detection_confidence: float,
        knowledge_confidence: float,
    ) -> float:
        """Calculate a weighted confidence score in [0.0, 1.0]."""
        evidence_strength = self._compute_evidence_strength(evidence)

        score = (
            self._config.confidence_weight_detection * detection_confidence
            + self._config.confidence_weight_knowledge * knowledge_confidence
            + self._config.confidence_weight_evidence * evidence_strength
        )

        clamped = max(0.0, min(1.0, score))
        logger.debug(
            "Confidence: detection=%.2f knowledge=%.2f evidence=%.2f -> %.2f",
            detection_confidence,
            knowledge_confidence,
            evidence_strength,
            clamped,
        )
        return clamped

    def _compute_evidence_strength(self, evidence: list[Evidence]) -> float:
        """Return the mean weight across all evidence items.

        Returns ``0.0`` when the list is empty.
        """
        if not evidence:
            return 0.0
        try:
            total = sum(e.weight for e in evidence)
            return total / len(evidence)
        except (TypeError, ValueError) as exc:
            raise ScoringError(f"Failed to compute evidence strength: {exc}") from exc
