"""Abstract interfaces for every component in the Fusion Engine pipeline.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from backend.contracts.models.analysis import EnemyAnalysis, FriendlyAnalysis, TerrainAnalysis
from backend.contracts.models.fusion import FusionResult, ThreatAssessment

# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CollectedIntelligence:
    """Aggregated intelligence from all three knowledge modules."""

    friendly: FriendlyAnalysis
    enemy: EnemyAnalysis
    terrain: TerrainAnalysis


@dataclass(frozen=True)
class ValidationResult:
    """Result of validating collected intelligence."""

    valid: bool
    issues: list[str]


@dataclass(frozen=True)
class CorrelatedEvidence:
    """Correlated findings across intelligence domains."""

    correlations: list[str]
    supporting_evidence: list[str]


@dataclass(frozen=True)
class ConflictRecord:
    """Identified and resolved conflicts between intelligence sources."""

    conflicts: list[str]
    resolutions: list[str]


@dataclass(frozen=True)
class SituationReport:
    """The Common Operational Picture built from fused intelligence."""

    summary: str
    supporting_evidence: list[str]
    key_observations: list[str]
    operational_context: str


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class FusionEngineInterface(ABC):
    """Public contract for the Fusion Engine.

    Implementations coordinate the full pipeline from domain analyses
    to a unified ``FusionResult`` and ``ThreatAssessment``.
    """

    @abstractmethod
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

    @abstractmethod
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


class CollectorInterface(ABC):
    """Contract for collecting intelligence from all domain modules.

    Aggregates the three analyses into a single structure for
    downstream processing.  Performs no validation, correlation,
    or confidence calculations.
    """

    @abstractmethod
    def collect(
        self,
        friendly: FriendlyAnalysis,
        enemy: EnemyAnalysis,
        terrain: TerrainAnalysis,
    ) -> CollectedIntelligence:
        """Collect the three domain analyses into a unified structure.

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
        CollectedIntelligence
            Aggregated intelligence for downstream processing.
        """


class ValidatorInterface(ABC):
    """Contract for validating collected intelligence.

    Checks for missing analyses, invalid contracts, incomplete
    intelligence, and required fields.
    """

    @abstractmethod
    def validate(
        self,
        collected: CollectedIntelligence,
    ) -> ValidationResult:
        """Validate collected intelligence for completeness and correctness.

        Parameters
        ----------
        collected:
            Aggregated intelligence from the collector.

        Returns
        -------
        ValidationResult
            Validation status and list of issues.
        """


class CorrelationEngineInterface(ABC):
    """Contract for correlating intelligence across domains.

    Finds relationships between friendly, enemy, and terrain
    analyses.  Identifies supporting evidence and builds a
    unified intelligence picture.
    """

    @abstractmethod
    def correlate(
        self,
        collected: CollectedIntelligence,
    ) -> CorrelatedEvidence:
        """Correlate intelligence across all three domains.

        Parameters
        ----------
        collected:
            Aggregated intelligence from the collector.

        Returns
        -------
        CorrelatedEvidence
            Correlated findings and supporting evidence.
        """


class ConflictResolverInterface(ABC):
    """Contract for detecting and resolving intelligence conflicts.

    Identifies situations where intelligence sources disagree
    and produces a resolved assessment.
    """

    @abstractmethod
    def resolve(
        self,
        collected: CollectedIntelligence,
        evidence: CorrelatedEvidence,
    ) -> ConflictRecord:
        """Detect and resolve conflicts between intelligence sources.

        Parameters
        ----------
        collected:
            Aggregated intelligence from the collector.
        evidence:
            Correlated evidence from the correlation engine.

        Returns
        -------
        ConflictRecord
            Identified conflicts and their resolutions.
        """


class SituationBuilderInterface(ABC):
    """Contract for building the Common Operational Picture.

    Produces a situation summary, supporting evidence,
    operational context, and key observations.
    """

    @abstractmethod
    def build(
        self,
        collected: CollectedIntelligence,
        evidence: CorrelatedEvidence,
        conflicts: ConflictRecord,
    ) -> SituationReport:
        """Build the Common Operational Picture from all intelligence.

        Parameters
        ----------
        collected:
            Aggregated intelligence from the collector.
        evidence:
            Correlated evidence from the correlation engine.
        conflicts:
            Resolved conflicts from the conflict resolver.

        Returns
        -------
        SituationReport
            The Common Operational Picture.
        """


class ConfidenceScorerInterface(ABC):
    """Contract for calculating combined fusion confidence.

    Consumes all three domain confidences and a correlation
    confidence to produce a single combined score.
    """

    @abstractmethod
    def score(
        self,
        friendly_confidence: float,
        enemy_confidence: float,
        terrain_confidence: float,
        correlation_confidence: float,
    ) -> float:
        """Calculate a combined confidence score in [0.0, 1.0].

        Parameters
        ----------
        friendly_confidence:
            Confidence from the friendly analysis [0, 1].
        enemy_confidence:
            Confidence from the enemy analysis [0, 1].
        terrain_confidence:
            Confidence from the terrain analysis [0, 1].
        correlation_confidence:
            Confidence derived from cross-domain correlation [0, 1].

        Returns
        -------
        float
            Combined confidence score clamped to [0.0, 1.0].
        """
