"""Module-specific exceptions for the Fusion Engine pipeline."""


class FusionError(RuntimeError):
    """Base exception for all Fusion Engine errors."""


class ValidationError(FusionError):
    """Raised when intelligence validation fails."""


class CorrelationError(FusionError):
    """Raised when intelligence correlation encounters an error."""


class ConflictResolutionError(FusionError):
    """Raised when conflict resolution fails."""


class SituationBuildError(FusionError):
    """Raised when situation building fails."""
