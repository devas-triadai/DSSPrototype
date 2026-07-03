"""Abstract interfaces for every component in the Terrain Knowledge pipeline.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from backend.contracts.models.analysis import TerrainAnalysis
from backend.contracts.models.detection import DetectionResult

# ---------------------------------------------------------------------------
# Internal data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TerrainData:
    """Raw terrain information retrieved for a location or area.

    Fields are intentionally optional to support sparse data from
    heterogeneous sources (GIS layers, DEM rasters, satellite imagery,
    offline maps, etc.).
    """

    source: str
    terrain_type: str | None = None
    elevation: float | None = None
    features: list[str] = field(default_factory=list)
    road_network: list[str] = field(default_factory=list)
    water_bodies: list[str] = field(default_factory=list)
    vegetation: list[str] = field(default_factory=list)
    obstacles: list[str] = field(default_factory=list)
    slope: float | None = None
    soil_type: str | None = None
    weather: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TerrainFeature:
    """An identified terrain feature with explainable attributes."""

    feature_type: str
    description: str
    confidence: float
    source: str


@dataclass(frozen=True)
class MobilityAssessment:
    """Assessment of terrain mobility and movement constraints."""

    road_access: bool
    mobility_rating: str
    terrain_difficulty: str
    obstacles: list[str]
    description: str


@dataclass(frozen=True)
class VisibilityAssessment:
    """Assessment of visibility conditions in the area."""

    visibility: str
    observation: str
    cover: str
    concealment: str
    description: str


@dataclass(frozen=True)
class RetrievalResult:
    """The output of a retrieval operation against a terrain source."""

    items: list[TerrainData] = field(default_factory=list)
    query_time_ms: float = 0.0


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class TerrainKnowledgeInterface(ABC):
    """Public contract for the Terrain Knowledge module.

    Implementations coordinate the full pipeline from
    ``DetectionResult`` to ``TerrainAnalysis``.
    """

    @abstractmethod
    async def analyze_terrain(self, detection: DetectionResult) -> TerrainAnalysis:
        """Analyse terrain features, visibility, and accessibility.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        TerrainAnalysis
            Terrain classification, features, and mobility assessment.
        """


class RetrieverInterface(ABC):
    """Contract for any terrain information retrieval source.

    Future implementations may back this interface with GIS services,
    DEM rasters, satellite tile servers, offline map databases, or
    GeoJSON files.
    """

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Query a terrain source and return matching data.

        Parameters
        ----------
        query:
            A location query (e.g. coordinates, place name, AOI).
        context:
            Optional contextual metadata (e.g. bounding box, zoom level).

        Returns
        -------
        RetrievalResult
            Terrain data items matched by the query.
        """


class TerrainEngineInterface(ABC):
    """Contract for the terrain reasoning engine.

    Receives a ``DetectionResult`` and orchestrates retrieval, feature
    extraction, mobility analysis, visibility analysis, and scoring to
    produce an assessment.  Does **not** perform retrieval or scoring
    itself.
    """

    @abstractmethod
    async def analyze(
        self,
        detection: DetectionResult,
        retriever: RetrieverInterface,
    ) -> TerrainAnalysis:
        """Analyse terrain using a detection result and the given retriever.

        Parameters
        ----------
        detection:
            Vision-detection output to analyse.
        retriever:
            Terrain source to query.

        Returns
        -------
        TerrainAnalysis
            Terrain classification, features, and mobility assessment.
        """


class TerrainFeatureBuilderInterface(ABC):
    """Contract for building explainable terrain features.

    Extracts structured ``TerrainFeature`` objects from raw
    ``TerrainData``.  No confidence, mobility, or visibility
    calculations are performed here.
    """

    @abstractmethod
    def build_features(
        self,
        data: list[TerrainData],
    ) -> list[TerrainFeature]:
        """Create a list of terrain features from retrieved data.

        Parameters
        ----------
        data:
            Raw terrain data items from the retriever.

        Returns
        -------
        list[TerrainFeature]
            Explainable terrain features (may be empty).
        """


class MobilityAnalyzerInterface(ABC):
    """Contract for analysing terrain mobility.

    Evaluates road access, terrain difficulty, obstacles, and movement
    constraints.  No confidence scoring is performed.
    """

    @abstractmethod
    def analyze(
        self,
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> MobilityAssessment:
        """Assess mobility conditions from terrain features and data.

        Parameters
        ----------
        features:
            Terrain features extracted from retrieved data.
        data:
            Raw terrain data (for additional context).

        Returns
        -------
        MobilityAssessment
            Mobility assessment including road access and difficulty.
        """


class VisibilityAnalyzerInterface(ABC):
    """Contract for analysing visibility and cover.

    Evaluates observation quality, cover, concealment, and line of
    sight.  No confidence scoring is performed.
    """

    @abstractmethod
    def analyze(
        self,
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> VisibilityAssessment:
        """Assess visibility conditions from terrain features and data.

        Parameters
        ----------
        features:
            Terrain features extracted from retrieved data.
        data:
            Raw terrain data (for additional context).

        Returns
        -------
        VisibilityAssessment
            Visibility assessment including cover and concealment.
        """


class ConfidenceScorerInterface(ABC):
    """Contract for calculating final terrain analysis confidence.

    Consumes detection confidence, terrain source confidence, and
    feature evidence strength to produce a single score.  The
    algorithm is replaceable.
    """

    @abstractmethod
    def score(
        self,
        detection_confidence: float,
        terrain_confidence: float,
        evidence_confidence: float,
    ) -> float:
        """Calculate a confidence score in [0.0, 1.0].

        Parameters
        ----------
        detection_confidence:
            The CV model's confidence for this detection [0, 1].
        terrain_confidence:
            Aggregate confidence of terrain source data [0, 1].
        evidence_confidence:
            Confidence derived from extracted feature quality [0, 1].

        Returns
        -------
        float
            Final confidence score clamped to [0.0, 1.0].
        """
