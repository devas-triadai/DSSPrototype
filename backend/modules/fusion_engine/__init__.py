"""Fusion Engine.

Merges intelligence from all knowledge domains into a unified
battlespace picture.  Correlates multi-source data, resolves
conflicts, and produces a single coherent assessment with
combined confidence scoring.
"""

from backend.modules.fusion_engine.collector import Collector
from backend.modules.fusion_engine.confidence_scorer import ConfidenceScorer
from backend.modules.fusion_engine.config import FusionConfig, fusion_config
from backend.modules.fusion_engine.conflict_resolver import ConflictResolver
from backend.modules.fusion_engine.correlation_engine import CorrelationEngine
from backend.modules.fusion_engine.exceptions import (
    ConflictResolutionError,
    CorrelationError,
    FusionError,
    SituationBuildError,
    ValidationError,
)
from backend.modules.fusion_engine.interfaces import (
    CollectedIntelligence,
    CollectorInterface,
    ConfidenceScorerInterface,
    ConflictRecord,
    ConflictResolverInterface,
    CorrelatedEvidence,
    CorrelationEngineInterface,
    FusionEngineInterface,
    SituationBuilderInterface,
    SituationReport,
    ValidationResult,
    ValidatorInterface,
)
from backend.modules.fusion_engine.service import FusionService
from backend.modules.fusion_engine.situation_builder import SituationBuilder
from backend.modules.fusion_engine.validator import Validator

__all__ = [
    # Config
    "FusionConfig",
    "fusion_config",
    # Service
    "FusionService",
    # Engine components
    "Collector",
    "Validator",
    "CorrelationEngine",
    "ConflictResolver",
    "SituationBuilder",
    "ConfidenceScorer",
    # Interfaces
    "FusionEngineInterface",
    "CollectorInterface",
    "ValidatorInterface",
    "CorrelationEngineInterface",
    "ConflictResolverInterface",
    "SituationBuilderInterface",
    "ConfidenceScorerInterface",
    # Types
    "CollectedIntelligence",
    "ValidationResult",
    "CorrelatedEvidence",
    "ConflictRecord",
    "SituationReport",
    # Exceptions
    "FusionError",
    "ValidationError",
    "CorrelationError",
    "ConflictResolutionError",
    "SituationBuildError",
]
