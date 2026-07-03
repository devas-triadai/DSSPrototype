"""Abstract interfaces for every component in the Enemy Knowledge pipeline.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.contracts.models.analysis import EnemyAnalysis
from backend.contracts.models.detection import DetectedObject, DetectionResult

# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KnowledgeItem:
    """A single piece of intelligence about an enemy unit or asset.

    Fields are intentionally optional to support sparse data from
    heterogeneous sources (threat libraries, equipment catalogues,
    open-source intelligence, etc.).
    """

    source: str
    unit_id: str | None = None
    unit_name: str | None = None
    equipment: list[str] = field(default_factory=list)
    characteristics: list[str] = field(default_factory=list)
    markings: list[str] = field(default_factory=list)
    country: str | None = None
    capabilities: list[str] = field(default_factory=list)
    threat_indicators: list[str] = field(default_factory=list)
    tactical_role: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Evidence:
    """An explainable piece of evidence linking a detection to enemy intelligence."""

    evidence_type: str
    description: str
    matched_attribute: str
    knowledge_source: str
    weight: float


@dataclass(frozen=True)
class RetrievalResult:
    """The output of a retrieval operation against an intelligence source."""

    items: list[KnowledgeItem] = field(default_factory=list)
    query_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class EnemyKnowledgeInterface(ABC):
    """Public contract for the Enemy Knowledge module.

    Implementations coordinate the full pipeline from
    ``DetectionResult`` to ``EnemyAnalysis``.
    """

    @abstractmethod
    async def analyze_enemy(self, detection: DetectionResult) -> EnemyAnalysis:
        """Analyse a detection result for enemy-force matches.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        EnemyAnalysis
            Assessment of enemy-force matches, possible equipment, and confidence.
        """


class RetrieverInterface(ABC):
    """Contract for any intelligence retrieval source.

    Future implementations may back this interface with vector
    databases, military manuals, equipment catalogues, threat
    libraries, or remote APIs.  No implementation may assume a
    hardcoded intelligence source.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Query an intelligence source and return matching items.

        Parameters
        ----------
        query:
            A natural-language or structured query string.
        context:
            Optional contextual metadata (e.g. AOI, threat level).

        Returns
        -------
        RetrievalResult
            Intelligence items matched by the query.
        """


class KnowledgeEngineInterface(ABC):
    """Contract for the intelligence reasoning engine.

    Receives a ``DetectionResult`` and orchestrates retrieval,
    filtering, evidence construction, and scoring to produce an
    assessment.  Does **not** perform retrieval or scoring itself.
    """

    @abstractmethod
    async def analyze(
        self,
        detection: DetectionResult,
        retriever: RetrieverInterface,
    ) -> EnemyAnalysis:
        """Analyse a detection result using the given retriever.

        Parameters
        ----------
        detection:
            Vision-detection output to analyse.
        retriever:
            Intelligence source to query.

        Returns
        -------
        EnemyAnalysis
            Assessment of enemy-force matches, possible equipment, and confidence.
        """


class EvidenceBuilderInterface(ABC):
    """Contract for constructing explainable evidence.

    Evidence links specific attributes of a detected object to
    matching attributes in an intelligence item.  No confidence
    calculations are performed here.
    """

    @abstractmethod
    def build_evidence(
        self,
        obj: DetectedObject,
        knowledge: list[KnowledgeItem],
    ) -> list[Evidence]:
        """Create a list of evidence items for a single detected object.

        Parameters
        ----------
        obj:
            A single detected object from the vision pipeline.
        knowledge:
            Intelligence items retrieved for this object.

        Returns
        -------
        list[Evidence]
            Explainable evidence items (may be empty).
        """


class ConfidenceScorerInterface(ABC):
    """Contract for calculating final confidence.

    Consumes evidence strength, detection confidence, and intelligence
    source confidence to produce a single score.  The algorithm is
    replaceable — swap implementations without changing the pipeline.
    """

    @abstractmethod
    def score(
        self,
        evidence: list[Evidence],
        detection_confidence: float,
        knowledge_confidence: float,
    ) -> float:
        """Calculate a confidence score in [0.0, 1.0].

        Parameters
        ----------
        evidence:
            Evidence items linking the detection to enemy intelligence.
        detection_confidence:
            The CV model's confidence for this detection [0, 1].
        knowledge_confidence:
            Aggregate confidence of matched intelligence items [0, 1].

        Returns
        -------
        float
            Final confidence score clamped to [0.0, 1.0].
        """
