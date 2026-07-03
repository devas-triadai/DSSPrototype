"""Confidence Adjuster.

Receives detector confidence and ontology confidence, and produces
an adjusted semantic confidence that never exceeds the detector confidence.
"""

import logging

from backend.modules.knowledge.ontology.interfaces import ConfidenceAdjusterInterface

logger = logging.getLogger("dss.ontology.confidence_adjuster")


class ConfidenceAdjuster(ConfidenceAdjusterInterface):
    """Adjusts confidence based on mapping quality and depth."""

    def __init__(
        self,
        depth_penalty: float = 0.95,
        synonym_bonus: float = 1.0,
        alias_penalty: float = 0.92,
        category_penalty: float = 0.90,
    ) -> None:
        self._depth_penalty = depth_penalty
        self._synonym_bonus = synonym_bonus
        self._alias_penalty = alias_penalty
        self._category_penalty = category_penalty

    def adjust(
        self,
        detector_confidence: float,
        ontology_confidence: float,
        mapping_depth: int,
    ) -> float:
        """Return adjusted confidence bounded by detector confidence.

        Args:
            detector_confidence: The original detector confidence [0, 1].
            ontology_confidence: Base confidence from the ontology entry [0, 1].
            mapping_depth: Number of hops from original label to concept (0 = direct).

        Returns:
            Adjusted confidence in [0, 1], never exceeding detector_confidence.
        """
        # Apply depth penalty
        depth_multiplier = self._depth_penalty ** mapping_depth
        adjusted = ontology_confidence * depth_multiplier

        # Ensure we never exceed detector confidence
        final = min(adjusted, detector_confidence)
        final = max(0.0, min(1.0, final))  # Clamp to [0, 1]

        logger.debug(
            "ConfidenceAdjuster | detector=%.3f ontology=%.3f depth=%d -> adjusted=%.3f",
            detector_confidence,
            ontology_confidence,
            mapping_depth,
            final,
        )
        return round(final, 3)

    def adjust_concept_confidence(
        self,
        detector_confidence: float,
        concept_confidence: float,
        source: str,
    ) -> float:
        """Adjust confidence based on mapping source type."""
        source_multiplier = {
            "direct": self._synonym_bonus,
            "synonym": self._synonym_bonus,
            "alias": self._alias_penalty,
            "category": self._category_penalty,
            "military_equivalent": self._alias_penalty,
            "civilian_equivalent": self._alias_penalty,
        }.get(source, 1.0)

        adjusted = concept_confidence * source_multiplier
        final = min(adjusted, detector_confidence)
        return round(max(0.0, min(1.0, final)), 3)
