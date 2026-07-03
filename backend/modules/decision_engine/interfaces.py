"""Abstract interfaces for every component in the Decision Engine pipeline.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from backend.contracts.enums.core import ThreatLevel
from backend.contracts.models.decision import CommanderDecision, DecisionRecommendation
from backend.contracts.models.fusion import FusionResult, ThreatAssessment

# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SituationContext:
    """Structured evaluation of the operational situation."""

    threat_level: ThreatLevel
    threat_confidence: float
    fusion_confidence: float
    has_enemy: bool
    has_friendly: bool
    terrain_summary: str
    key_observations: list[str] = field(default_factory=list)
    severity: str = "unknown"


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class DecisionEngineInterface(ABC):
    """Public contract for the Decision Engine.

    Implementations coordinate the full pipeline from fused
    intelligence to a ranked ``DecisionRecommendation``.
    """

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

    @abstractmethod
    async def process_decision(self, decision: CommanderDecision) -> None:
        """Record and act on the commander's final decision.

        Parameters
        ----------
        decision:
            The commander's chosen course of action and metadata.
        """


class SituationEvaluatorInterface(ABC):
    """Contract for evaluating the operational situation.

    Receives fused intelligence and threat assessment to produce
    a structured situation context.  No recommendations.
    """

    @abstractmethod
    def evaluate(
        self,
        fusion: FusionResult,
        threat: ThreatAssessment,
    ) -> SituationContext:
        """Evaluate the operational situation from fused intelligence.

        Parameters
        ----------
        fusion:
            The fused intelligence picture.
        threat:
            The assessed threat level and confidence.

        Returns
        -------
        SituationContext
            Structured evaluation of the current situation.
        """


class COAGeneratorInterface(ABC):
    """Contract for generating Courses of Action.

    Produces a list of possible actions based on the situation
    and threat assessment.  The generator must be configurable
    and contain no hardcoded tactical doctrine.
    """

    @abstractmethod
    def generate(
        self,
        situation: SituationContext,
        threat: ThreatAssessment,
    ) -> list[str]:
        """Generate possible courses of action.

        Parameters
        ----------
        situation:
            Structured situation context.
        threat:
            The assessed threat level and confidence.

        Returns
        -------
        list[str]
            Ordered list of recommended actions (may be empty).
        """


class PriorityAnalyzerInterface(ABC):
    """Contract for assigning recommendation priority.

    Priority is based on threat level, confidence, and situation
    severity.  No recommendations.
    """

    @abstractmethod
    def analyze(
        self,
        situation: SituationContext,
        threat: ThreatAssessment,
    ) -> int:
        """Assign a priority score (1 = highest, 5 = lowest).

        Parameters
        ----------
        situation:
            Structured situation context.
        threat:
            The assessed threat level and confidence.

        Returns
        -------
        int
            Priority level from 1 (highest) to 5 (lowest).
        """


class RecommendationBuilderInterface(ABC):
    """Contract for building the final DecisionRecommendation.

    Combines situation context, COAs, priority, and confidence
    into a single explainable recommendation.
    """

    @abstractmethod
    def build(
        self,
        situation: SituationContext,
        actions: list[str],
        priority: int,
        confidence: float,
    ) -> DecisionRecommendation:
        """Build a DecisionRecommendation from all analysis results.

        Parameters
        ----------
        situation:
            Structured situation context.
        actions:
            Ordered list of recommended actions.
        priority:
            Assigned priority (1-5).
        confidence:
            Combined recommendation confidence.

        Returns
        -------
        DecisionRecommendation
            The final recommendation with reasoning.
        """


class ConfidenceScorerInterface(ABC):
    """Contract for calculating overall recommendation confidence.

    Consumes fusion confidence, threat confidence, and situation
    confidence to produce a single score.  Replaceable.
    """

    @abstractmethod
    def score(
        self,
        fusion_confidence: float,
        threat_confidence: float,
        situation_confidence: float,
    ) -> float:
        """Calculate a combined confidence score in [0.0, 1.0].

        Parameters
        ----------
        fusion_confidence:
            Confidence from the fusion engine [0, 1].
        threat_confidence:
            Confidence from the threat assessment [0, 1].
        situation_confidence:
            Confidence derived from situation evaluation [0, 1].

        Returns
        -------
        float
            Combined confidence score clamped to [0.0, 1.0].
        """
