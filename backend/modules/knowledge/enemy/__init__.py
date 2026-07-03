"""Enemy Knowledge Module.

Analyses detected objects for matches against known enemy forces
using a replaceable intelligence-retrieval pipeline.  Contains zero
AI, RAG, or vector-database logic.
"""

from backend.modules.knowledge.enemy.confidence_scorer import ConfidenceScorer
from backend.modules.knowledge.enemy.config import EnemyConfig, enemy_config
from backend.modules.knowledge.enemy.evidence_builder import EvidenceBuilder
from backend.modules.knowledge.enemy.exceptions import (
    EvidenceError,
    KnowledgeError,
    RetrievalError,
    ScoringError,
)
from backend.modules.knowledge.enemy.interfaces import (
    ConfidenceScorerInterface,
    EnemyKnowledgeInterface,
    Evidence,
    EvidenceBuilderInterface,
    KnowledgeEngineInterface,
    KnowledgeItem,
    RetrievalResult,
    RetrieverInterface,
)
from backend.modules.knowledge.enemy.knowledge_engine import KnowledgeEngine
from backend.modules.knowledge.enemy.retriever import NullRetriever
from backend.modules.knowledge.enemy.service import EnemyKnowledgeService

__all__ = [
    # Config
    "EnemyConfig",
    "enemy_config",
    # Service
    "EnemyKnowledgeService",
    # Engine
    "KnowledgeEngine",
    "ConfidenceScorer",
    "EvidenceBuilder",
    "NullRetriever",
    # Interfaces
    "EnemyKnowledgeInterface",
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
