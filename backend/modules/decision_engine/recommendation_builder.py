"""Recommendation building for the Decision Engine.

Combines situation context, courses of action, priority, and
confidence into a single explainable DecisionRecommendation.
"""

import logging

from backend.contracts.models.decision import DecisionRecommendation
from backend.modules.decision_engine.interfaces import (
    RecommendationBuilderInterface,
    SituationContext,
)

logger = logging.getLogger("dss.decision.recommendation_builder")


class RecommendationBuilder(RecommendationBuilderInterface):
    """Builds a ``DecisionRecommendation`` from all analysis results.

    Produces:
      - Ordered list of recommended actions
      - Priority (1-5)
      - Explainable reason string synthesised from situation context
    """

    def build(
        self,
        situation: SituationContext,
        actions: list[str],
        priority: int,
        confidence: float,
    ) -> DecisionRecommendation:
        """Build a DecisionRecommendation from all analysis results."""
        reason = self._build_reason(situation, actions, priority, confidence)

        return DecisionRecommendation(
            recommended_actions=actions,
            priority=priority,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_reason(
        situation: SituationContext,
        actions: list[str],
        priority: int,
        confidence: float,
    ) -> str:
        """Build an explainable reason string."""
        parts: list[str] = []

        parts.append(
            f"Situation: threat={situation.threat_level.value}, "
            f"severity={situation.severity}"
        )

        if situation.has_enemy:
            parts.append("enemy present")
        if situation.has_friendly:
            parts.append("friendly present")

        actions_str = "; ".join(actions)
        parts.append(f"Recommended: {actions_str}")

        parts.append(f"Priority {priority}/5 (confidence={confidence:.2f})")

        return " | ".join(parts)
