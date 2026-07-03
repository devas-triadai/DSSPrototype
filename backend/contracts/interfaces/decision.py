"""Interface for the Decision Support Agent."""

from abc import ABC, abstractmethod

from backend.contracts.models.decision import CommanderDecision, DecisionRecommendation
from backend.contracts.models.fusion import FusionResult, ThreatAssessment


class DecisionModule(ABC):
    """Contract for generating course-of-action recommendations."""

    @abstractmethod
    async def generate_recommendations(
        self,
        threat: ThreatAssessment,
        fusion: FusionResult,
    ) -> DecisionRecommendation:
        """Produce a ranked recommendation based on threat and fused intelligence.

        Parameters
        ----------
        threat:
            The assessed threat level and confidence.
        fusion:
            The fused intelligence picture.

        Returns
        -------
        DecisionRecommendation
            Recommended actions, priority, and rationale.
        """
        ...

    @abstractmethod
    async def process_decision(self, decision: CommanderDecision) -> None:
        """Record and act on the commander's final decision.

        Parameters
        ----------
        decision:
            The commander's chosen course of action and metadata.
        """
        ...
