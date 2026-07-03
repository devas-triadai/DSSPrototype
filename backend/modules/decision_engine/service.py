"""Public entry point for the Decision Engine.

Coordinates the full decision-recommendation pipeline while
containing zero situation evaluation, COA generation, priority
analysis, recommendation building, or scoring logic.
"""

import logging
from typing import Any

from backend.contracts.interfaces.decision import DecisionModule
from backend.contracts.models.decision import CommanderDecision, DecisionRecommendation
from backend.contracts.models.fusion import FusionResult, ThreatAssessment
from backend.modules.decision_engine.coa_generator import COAGenerator
from backend.modules.decision_engine.confidence_scorer import ConfidenceScorer
from backend.modules.decision_engine.config import decision_config
from backend.modules.decision_engine.interfaces import (
    COAGeneratorInterface,
    ConfidenceScorerInterface,
    PriorityAnalyzerInterface,
    RecommendationBuilderInterface,
    SituationEvaluatorInterface,
)
from backend.modules.decision_engine.priority_analyzer import PriorityAnalyzer
from backend.modules.decision_engine.recommendation_builder import (
    RecommendationBuilder,
)
from backend.modules.decision_engine.situation_evaluator import SituationEvaluator

logger = logging.getLogger("dss.decision.service")


class DecisionService(DecisionModule):
    """Orchestrates the end-to-end decision recommendation pipeline.

    Pipeline steps:
        1. Evaluate the operational situation.
        2. Generate courses of action.
        3. Assign recommendation priority.
        4. Score recommendation confidence.
        5. Build the final DecisionRecommendation.

    Dependencies are injected via the constructor; sensible defaults
    are provided for every component.
    """

    def __init__(
        self,
        situation_evaluator: SituationEvaluatorInterface | None = None,
        coa_generator: COAGeneratorInterface | None = None,
        priority_analyzer: PriorityAnalyzerInterface | None = None,
        recommendation_builder: RecommendationBuilderInterface | None = None,
        confidence_scorer: ConfidenceScorerInterface | None = None,
        config: Any | None = None,
    ) -> None:
        self._situation_evaluator = situation_evaluator or SituationEvaluator()
        self._coa_generator = coa_generator or COAGenerator()
        self._priority_analyzer = priority_analyzer or PriorityAnalyzer()
        self._recommendation_builder = recommendation_builder or RecommendationBuilder()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._config = config or decision_config

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
        logger.info(
            "Generating recommendations (threat=%s, fusion_conf=%.2f)",
            threat.threat_level.value,
            fusion.combined_confidence,
        )

        situation = self._situation_evaluator.evaluate(fusion, threat)

        actions = self._coa_generator.generate(situation, threat)

        priority = self._priority_analyzer.analyze(situation, threat)

        situation_confidence = self._compute_situation_confidence(situation)
        recommendation_confidence = self._confidence_scorer.score(
            fusion_confidence=fusion.combined_confidence,
            threat_confidence=threat.confidence,
            situation_confidence=situation_confidence,
        )

        recommendation = self._recommendation_builder.build(
            situation=situation,
            actions=actions,
            priority=priority,
            confidence=recommendation_confidence,
        )

        logger.info(
            "Recommendation %s: priority %d, %d action(s), confidence=%.2f",
            recommendation.recommendation_id,
            recommendation.priority,
            len(recommendation.recommended_actions),
            recommendation_confidence,
        )

        return recommendation

    async def process_decision(self, decision: CommanderDecision) -> None:
        """Record and act on the commander's final decision.

        In the current framework this logs the decision.  Future
        implementations may forward it to a doctrine engine, rules
        engine, or human-approval workflow.

        Parameters
        ----------
        decision:
            The commander's chosen course of action and metadata.
        """
        logger.info(
            "Commander decision recorded: operator=%s, action='%s', remarks='%s'",
            decision.operator_name,
            decision.decision,
            decision.remarks or "",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_situation_confidence(situation: Any) -> float:
        """Derive a situation confidence from context quality."""
        base = decision_config.default_confidence
        if situation.has_enemy or situation.has_friendly:
            base += 0.2
        if situation.key_observations:
            base += min(len(situation.key_observations) * 0.05, 0.2)
        if situation.threat_level.value != "unknown":
            base += 0.1
        return min(base, 1.0)
