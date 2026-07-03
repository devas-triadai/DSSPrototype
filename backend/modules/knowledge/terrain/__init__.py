"""Terrain Knowledge Module.

Analyses terrain characteristics from detection data using a
replaceable retrieval pipeline.  Contains zero GIS, mapping,
RAG, or AI logic.
"""

from backend.modules.knowledge.terrain.confidence_scorer import ConfidenceScorer
from backend.modules.knowledge.terrain.config import TerrainConfig, terrain_config
from backend.modules.knowledge.terrain.exceptions import (
    KnowledgeError,
    MobilityError,
    RetrievalError,
    ScoringError,
    TerrainError,
    VisibilityError,
)
from backend.modules.knowledge.terrain.interfaces import (
    ConfidenceScorerInterface,
    MobilityAnalyzerInterface,
    MobilityAssessment,
    RetrievalResult,
    RetrieverInterface,
    TerrainData,
    TerrainEngineInterface,
    TerrainFeature,
    TerrainFeatureBuilderInterface,
    TerrainKnowledgeInterface,
    VisibilityAnalyzerInterface,
    VisibilityAssessment,
)
from backend.modules.knowledge.terrain.mobility_analyzer import MobilityAnalyzer
from backend.modules.knowledge.terrain.retriever import NullRetriever
from backend.modules.knowledge.terrain.service import TerrainKnowledgeService
from backend.modules.knowledge.terrain.terrain_engine import TerrainEngine
from backend.modules.knowledge.terrain.terrain_feature_builder import (
    TerrainFeatureBuilder,
)
from backend.modules.knowledge.terrain.visibility_analyzer import VisibilityAnalyzer

__all__ = [
    # Config
    "TerrainConfig",
    "terrain_config",
    # Service
    "TerrainKnowledgeService",
    # Engine
    "TerrainEngine",
    "ConfidenceScorer",
    "TerrainFeatureBuilder",
    "MobilityAnalyzer",
    "VisibilityAnalyzer",
    "NullRetriever",
    # Interfaces
    "TerrainKnowledgeInterface",
    "RetrieverInterface",
    "TerrainEngineInterface",
    "TerrainFeatureBuilderInterface",
    "MobilityAnalyzerInterface",
    "VisibilityAnalyzerInterface",
    "ConfidenceScorerInterface",
    # Types
    "TerrainData",
    "TerrainFeature",
    "MobilityAssessment",
    "VisibilityAssessment",
    "RetrievalResult",
    # Exceptions
    "KnowledgeError",
    "RetrievalError",
    "TerrainError",
    "MobilityError",
    "VisibilityError",
    "ScoringError",
]
