"""Decision Engine.

Generates course-of-action recommendations based on fused
intelligence and threat assessment.  Evaluates options,
assigns priorities, and supports commander decision-making.
Contains zero AI and never issues autonomous orders.
"""

from backend.modules.decision_engine.coa_generator import COAGenerator
from backend.modules.decision_engine.confidence_scorer import ConfidenceScorer
from backend.modules.decision_engine.config import DecisionConfig, decision_config
from backend.modules.decision_engine.exceptions import (
    COAGenerationError,
    DecisionError,
    PriorityError,
    RecommendationError,
    SituationEvaluationError,
)
from backend.modules.decision_engine.interfaces import (
    COAGeneratorInterface,
    ConfidenceScorerInterface,
    DecisionEngineInterface,
    PriorityAnalyzerInterface,
    RecommendationBuilderInterface,
    SituationContext,
    SituationEvaluatorInterface,
)
from backend.modules.decision_engine.priority_analyzer import PriorityAnalyzer
from backend.modules.decision_engine.recommendation_builder import (
    RecommendationBuilder,
)
from backend.modules.decision_engine.service import DecisionService
from backend.modules.decision_engine.situation_evaluator import SituationEvaluator

__all__ = [
    # Config
    "DecisionConfig",
    "decision_config",
    # Service
    "DecisionService",
    # Engine components
    "SituationEvaluator",
    "COAGenerator",
    "PriorityAnalyzer",
    "RecommendationBuilder",
    "ConfidenceScorer",
    # Interfaces
    "DecisionEngineInterface",
    "SituationEvaluatorInterface",
    "COAGeneratorInterface",
    "PriorityAnalyzerInterface",
    "RecommendationBuilderInterface",
    "ConfidenceScorerInterface",
    # Types
    "SituationContext",
    # Exceptions
    "DecisionError",
    "SituationEvaluationError",
    "COAGenerationError",
    "PriorityError",
    "RecommendationError",
]
