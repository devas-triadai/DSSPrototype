"""Friendly Knowledge Module.

Analyses detected objects for matches against known friendly forces
using a replaceable knowledge-retrieval pipeline.  Contains zero
AI, RAG, or vector-database logic.
"""

from backend.modules.knowledge.friendly.confidence_scorer import ConfidenceScorer
from backend.modules.knowledge.friendly.config import FriendlyConfig, friendly_config
from backend.modules.knowledge.friendly.evidence_builder import EvidenceBuilder
from backend.modules.knowledge.friendly.exceptions import (
    EvidenceError,
    KnowledgeError,
    RetrievalError,
    ScoringError,
)
from backend.modules.knowledge.friendly.interfaces import (
    ConfidenceScorerInterface,
    Evidence,
    EvidenceBuilderInterface,
    FriendlyKnowledgeInterface,
    KnowledgeEngineInterface,
    KnowledgeItem,
    RetrievalResult,
    RetrieverInterface,
)
from backend.modules.knowledge.friendly.knowledge_engine import KnowledgeEngine
from backend.modules.knowledge.friendly.retriever import NullRetriever
from backend.modules.knowledge.friendly.service import FriendlyKnowledgeService

__all__ = [
    # Config
    "FriendlyConfig",
    "friendly_config",
    # Service
    "FriendlyKnowledgeService",
    # Engine
    "KnowledgeEngine",
    "ConfidenceScorer",
    "EvidenceBuilder",
    "NullRetriever",
    # Interfaces
    "FriendlyKnowledgeInterface",
    "RetrieverInterface",
    "KnowledgeEngineInterface",
    "EvidenceBuilderInterface",
    "ConfidenceScorerInterface",
    # Types
    "KnowledgeItem",
    "Evidence",
    "RetrievalResult",
    # Exceptions
    "KnowledgeError",
    "RetrievalError",
    "EvidenceError",
    "ScoringError",
]
