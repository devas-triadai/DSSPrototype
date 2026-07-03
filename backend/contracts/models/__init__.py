"""Data-transfer objects shared between DSS modules.

All inter-module communication must use these strongly typed models.
"""

from backend.contracts.models.analysis import EnemyAnalysis, FriendlyAnalysis, TerrainAnalysis
from backend.contracts.models.decision import CommanderDecision, DecisionRecommendation
from backend.contracts.models.detection import DetectedObject, DetectionResult, ImageMetadata
from backend.contracts.models.fusion import FusionResult, ThreatAssessment
from backend.contracts.models.geometry import (
    AnnotationGeometry,
    BoundingBox,
    OrientedBBox,
    Polygon,
)

__all__ = [
    "BoundingBox",
    "OrientedBBox",
    "Polygon",
    "AnnotationGeometry",
    "ImageMetadata",
    "DetectedObject",
    "DetectionResult",
    "FriendlyAnalysis",
    "EnemyAnalysis",
    "TerrainAnalysis",
    "FusionResult",
    "ThreatAssessment",
    "DecisionRecommendation",
    "CommanderDecision",
]
