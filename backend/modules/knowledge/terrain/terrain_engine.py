"""Terrain reasoning engine for terrain assessment.

Orchestrates retrieval, feature extraction, mobility analysis,
visibility analysis, and confidence scoring to produce a
``TerrainAnalysis``.  Does **not** perform retrieval or scoring
itself — delegates to injected interfaces.
"""

import logging
from typing import Any

from backend.contracts.enums.core import TerrainType
from backend.contracts.models.analysis import TerrainAnalysis
from backend.contracts.models.detection import DetectionResult
from backend.modules.knowledge.terrain.confidence_scorer import ConfidenceScorer
from backend.modules.knowledge.terrain.config import terrain_config
from backend.modules.knowledge.terrain.interfaces import (
    ConfidenceScorerInterface,
    MobilityAnalyzerInterface,
    MobilityAssessment,
    RetrieverInterface,
    TerrainData,
    TerrainEngineInterface,
    TerrainFeature,
    TerrainFeatureBuilderInterface,
    VisibilityAnalyzerInterface,
    VisibilityAssessment,
)
from backend.modules.knowledge.terrain.mobility_analyzer import MobilityAnalyzer
from backend.modules.knowledge.terrain.terrain_feature_builder import (
    TerrainFeatureBuilder,
)
from backend.modules.knowledge.terrain.visibility_analyzer import VisibilityAnalyzer

logger = logging.getLogger("dss.knowledge.terrain.terrain_engine")


class TerrainEngine(TerrainEngineInterface):
    """Coordinates the terrain-assessment pipeline.

    Pipeline:
        1. Build query context from ``DetectionResult``.
        2. Retrieve terrain information for the detection area.
        3. Extract structured terrain features.
        4. Analyse mobility conditions.
        5. Analyse visibility conditions.
        6. Score final confidence.
        7. Assemble ``TerrainAnalysis``.
    """

    def __init__(
        self,
        feature_builder: TerrainFeatureBuilderInterface | None = None,
        mobility_analyzer: MobilityAnalyzerInterface | None = None,
        visibility_analyzer: VisibilityAnalyzerInterface | None = None,
        confidence_scorer: ConfidenceScorerInterface | None = None,
        config: Any | None = None,
    ) -> None:
        self._feature_builder = feature_builder or TerrainFeatureBuilder()
        self._mobility_analyzer = mobility_analyzer or MobilityAnalyzer()
        self._visibility_analyzer = visibility_analyzer or VisibilityAnalyzer()
        self._confidence_scorer = confidence_scorer or ConfidenceScorer()
        self._config = config or terrain_config

    async def analyze(
        self,
        detection: DetectionResult,
        retriever: RetrieverInterface,
    ) -> TerrainAnalysis:
        """Analyse terrain using a detection result and the given retriever."""
        query = self._build_query(detection)
        context = self._build_context(detection)

        try:
            result = await retriever.retrieve(query, context)
        except Exception as exc:
            logger.warning("Terrain retrieval failed: %s", exc)
            result = self._empty_retrieval_result()

        terrain_data = result.items
        features = self._feature_builder.build_features(terrain_data)

        mobility = self._mobility_analyzer.analyze(features, terrain_data)
        visibility = self._visibility_analyzer.analyze(features, terrain_data)

        terrain_type = self._classify_terrain(features, terrain_data)
        elevation = self._extract_elevation(terrain_data)
        nearby_features = self._collect_feature_names(features)

        feature_confidence = self._compute_evidence_confidence(features)
        avg_terrain_conf = self._average_terrain_confidence(terrain_data)
        avg_detection_conf = self._average_detection_confidence(detection)

        final_confidence = self._confidence_scorer.score(
            detection_confidence=avg_detection_conf,
            terrain_confidence=avg_terrain_conf,
            evidence_confidence=feature_confidence,
        )

        reason = self._build_reason(
            terrain_type,
            mobility,
            visibility,
            features,
            final_confidence,
        )

        return TerrainAnalysis(
            terrain_type=terrain_type,
            nearby_features=nearby_features,
            visibility=visibility.visibility,
            road_access=mobility.road_access,
            elevation=elevation,
            reason=reason,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_query(detection: DetectionResult) -> str:
        """Build a terrain query from detection object types."""
        type_labels = list({o.object_type.value for o in detection.objects})
        return " ".join(type_labels)

    @staticmethod
    def _build_context(detection: DetectionResult) -> dict[str, Any]:
        return {
            "image_id": detection.image_id,
            "timestamp": detection.timestamp.isoformat(),
            "object_count": len(detection.objects),
        }

    @staticmethod
    def _empty_retrieval_result() -> Any:
        from backend.modules.knowledge.terrain.interfaces import RetrievalResult
        return RetrievalResult(items=[], query_time_ms=0.0)

    @staticmethod
    def _classify_terrain(
        features: list[TerrainFeature],
        data: list[TerrainData],
    ) -> TerrainType:
        """Classify dominant terrain type from features and data."""
        type_counts: dict[str, int] = {}

        for feat in features:
            ft = feat.feature_type
            type_counts[ft] = type_counts.get(ft, 0) + 1

        for item in data:
            if item.terrain_type:
                tt = item.terrain_type.lower()
                type_counts[tt] = type_counts.get(tt, 0) + 2

        if not type_counts:
            return TerrainType.UNKNOWN

        dominant = max(type_counts, key=type_counts.get)

        mapping: dict[str, TerrainType] = {
            "urban": TerrainType.URBAN,
            "building": TerrainType.URBAN,
            "forest": TerrainType.FOREST,
            "vegetation": TerrainType.FOREST,
            "water": TerrainType.RIVER,
            "river": TerrainType.RIVER,
            "road": TerrainType.ROAD,
            "bridge": TerrainType.BRIDGE,
            "hill": TerrainType.HILL,
            "elevation": TerrainType.HILL,
            "valley": TerrainType.HILL,
            "open_field": TerrainType.OPEN_FIELD,
            "field": TerrainType.OPEN_FIELD,
            "soil": TerrainType.OPEN_FIELD,
        }
        return mapping.get(dominant, TerrainType.UNKNOWN)

    @staticmethod
    def _extract_elevation(data: list[TerrainData]) -> float | None:
        """Extract best elevation value from terrain data."""
        elevations = [d.elevation for d in data if d.elevation is not None]
        if not elevations:
            return None
        return max(elevations)

    @staticmethod
    def _collect_feature_names(features: list[TerrainFeature]) -> list[str]:
        """Collect unique, human-readable feature descriptions."""
        seen: set[str] = set()
        names: list[str] = []
        for f in features:
            key = f.feature_type
            if key not in seen:
                seen.add(key)
                names.append(key)
        return names

    @staticmethod
    def _compute_evidence_confidence(features: list[TerrainFeature]) -> float:
        """Compute a confidence score from the number and quality of features."""
        if not features:
            return 0.0
        total_conf = sum(f.confidence for f in features)
        return min(total_conf / len(features), 1.0)

    @staticmethod
    def _average_terrain_confidence(data: list[TerrainData]) -> float:
        """Return the mean confidence of terrain data sources."""
        if not data:
            return 0.0
        total = sum(d.confidence for d in data)
        return total / len(data)

    @staticmethod
    def _average_detection_confidence(detection: DetectionResult) -> float:
        """Return the mean detection confidence across all objects."""
        if not detection.objects:
            return 0.0
        total = sum(o.confidence for o in detection.objects)
        return total / len(detection.objects)

    @staticmethod
    def _build_reason(
        terrain_type: TerrainType,
        mobility: MobilityAssessment,
        visibility: VisibilityAssessment,
        features: list[TerrainFeature],
        confidence: float,
    ) -> str:
        """Build a human-readable reason string."""
        feat_count = len(features)
        return (
            f"Terrain classified as {terrain_type.value} (confidence={confidence:.2f}). "
            f"{feat_count} feature(s) identified. "
            f"{mobility.description} "
            f"{visibility.description}"
        )
