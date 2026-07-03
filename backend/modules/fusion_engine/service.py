"""Public entry point for the Fusion Engine.

Coordinates the full intelligence fusion pipeline while
containing zero correlation, conflict resolution, situation
building, or scoring logic.
"""

import logging
from typing import Any

from backend.contracts.enums.core import ThreatLevel
from backend.contracts.interfaces.fusion import FusionModule
from backend.contracts.models.analysis import EnemyAnalysis, FriendlyAnalysis, TerrainAnalysis
from backend.contracts.models.fusion import FusionResult, ThreatAssessment
from backend.modules.fusion_engine.collector import Collector
from backend.modules.fusion_engine.confidence_scorer import ConfidenceScorer
from backend.modules.fusion_engine.config import fusion_config
from backend.modules.fusion_engine.conflict_resolver import ConflictResolver
from backend.modules.fusion_engine.correlation_engine import CorrelationEngine
from backend.modules.fusion_engine.interfaces import (
    CollectorInterface,
    ConfidenceScorerInterface,
    ConflictResolverInterface,
    CorrelationEngineInterface,
    SituationBuilderInterface,
    ValidatorInterface,
)
from backend.modules.fusion_engine.situation_builder import SituationBuilder
from backend.modules.fusion_engine.validator import Validator

logger = logging.getLogger("dss.fusion.service")


class FusionService(FusionModule):
    """Orchestrates the end-to-end intelligence fusion pipeline.

    Pipeline steps:
        1. Collect intelligence from all three domain modules.
        2. Validate collected intelligence.
        3. Correlate intelligence across domains.
        4. Resolve conflicts between sources.
        5. Build the Common Operational Picture.
        6. Score combined confidence.

    Dependencies are injected via the constructor; sensible defaults
    are provided for every component.
    """

    def __init__(
        self,
        collector: CollectorInterface | None = None,
        validator: ValidatorInterface | None = None,
        correlation_engine: CorrelationEngineInterface | None = None,
        conflict_resolver: ConflictResolverInterface | None = None,
        situation_builder: SituationBuilderInterface | None = None,
        confidence_scorer: ConfidenceScorerInterface | None = None,
        config: Any | None = None,
    ) -> None:
        self._collector = collector or Collector()
        self._validator = validator or Validator()
        self._correlation_engine = correlation_engine or CorrelationEngine()
        self._conflict_resolver = conflict_resolver or ConflictResolver()
        self._situation_builder = situation_builder or SituationBuilder()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._config = config or fusion_config

    async def fuse_intelligence(
        self,
        friendly: FriendlyAnalysis,
        enemy: EnemyAnalysis,
        terrain: TerrainAnalysis,
    ) -> FusionResult:
        """Merge analyses from all three domain agents into a single fused picture.

        Parameters
        ----------
        friendly:
            Assessment from the Friendly Intelligence Agent.
        enemy:
            Assessment from the Enemy Intelligence Agent.
        terrain:
            Assessment from the Terrain Intelligence Agent.

        Returns
        -------
        FusionResult
            Combined confidence, summary, and supporting evidence.
        """
        logger.info("Fusing intelligence from all three domains")

        collected = self._collector.collect(friendly, enemy, terrain)

        validation = self._validator.validate(collected)
        if self._config.strict_validation and not validation.valid:
            logger.error("Intelligence validation failed: %s", validation.issues)

        evidence = self._correlation_engine.correlate(collected)
        conflicts = self._conflict_resolver.resolve(collected, evidence)
        situation = self._situation_builder.build(collected, evidence, conflicts)

        terrain_confidence = self._terrain_confidence(terrain)

        combined_confidence = self._confidence_scorer.score(
            friendly_confidence=friendly.confidence,
            enemy_confidence=enemy.confidence,
            terrain_confidence=terrain_confidence,
            correlation_confidence=self._correlation_confidence(evidence),
        )

        return FusionResult(
            combined_confidence=combined_confidence,
            summary=situation.summary,
            supporting_evidence=situation.supporting_evidence,
        )

    async def assess_threat(self, fusion: FusionResult) -> ThreatAssessment:
        """Derive a threat-level assessment from the fused intelligence.

        Parameters
        ----------
        fusion:
            The fused intelligence picture.

        Returns
        -------
        ThreatAssessment
            Severity level, confidence, and reasoning.
        """
        logger.info("Assessing threat from fused intelligence")

        threat_level = self._determine_threat_level(fusion)
        confidence = fusion.combined_confidence
        reason = self._build_threat_reason(threat_level, fusion)

        return ThreatAssessment(
            threat_level=threat_level,
            confidence=confidence,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _terrain_confidence(terrain: Any) -> float:
        """Derive a confidence proxy from terrain analysis fields."""
        base = fusion_config.default_confidence
        if terrain.terrain_type and terrain.visibility:
            return min(base + 0.2, 1.0)
        return base

    @staticmethod
    def _correlation_confidence(evidence: Any) -> float:
        """Derive a confidence score from correlation results."""
        if not evidence.correlations and not evidence.supporting_evidence:
            return 0.0
        correlation_count = len(evidence.correlations)
        evidence_count = len(evidence.supporting_evidence)
        raw = (correlation_count * 0.1 + evidence_count * 0.05)
        return min(raw, 1.0)

    @staticmethod
    def _determine_threat_level(fusion: FusionResult) -> ThreatLevel:
        """Determine threat level from the fused intelligence summary."""
        summary_lower = fusion.summary.lower()

        if "enemy presence" in summary_lower:
            return ThreatLevel.HIGH
        if "friendly presence" in summary_lower:
            return ThreatLevel.LOW
        if "conflict" in summary_lower:
            return ThreatLevel.MEDIUM
        return ThreatLevel.UNKNOWN

    @staticmethod
    def _build_threat_reason(
        threat_level: ThreatLevel,
        fusion: FusionResult,
    ) -> str:
        """Build a reason string for the threat assessment."""
        return (
            f"Threat assessed as {threat_level.value} "
            f"(combined confidence={fusion.combined_confidence:.2f}). "
            f"{fusion.summary[:200]}"
        )
