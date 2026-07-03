"""Module-specific exceptions for the Decision Engine pipeline."""


class DecisionError(RuntimeError):
    """Base exception for all Decision Engine errors."""


class SituationEvaluationError(DecisionError):
    """Raised when situation evaluation fails."""


class COAGenerationError(DecisionError):
    """Raised when course-of-action generation fails."""


class PriorityError(DecisionError):
    """Raised when priority assignment fails."""


class RecommendationError(DecisionError):
    """Raised when recommendation building fails."""
