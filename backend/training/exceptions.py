"""Module-specific exceptions for the Training Platform."""


class TrainingError(Exception):
    """Base exception for training-related errors."""


class ExperimentNotFoundError(TrainingError):
    """Raised when an experiment cannot be found."""


class ModelNotFoundError(TrainingError):
    """Raised when a model cannot be found in the registry."""


class CheckpointNotFoundError(TrainingError):
    """Raised when a checkpoint cannot be found."""


class ConfigurationError(TrainingError):
    """Raised when training configuration is invalid."""


class ExportError(TrainingError):
    """Raised when model export fails."""


class EvaluationError(TrainingError):
    """Raised when evaluation fails."""


class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted by a callback or early stopping."""


class DatasetNotReadyError(TrainingError):
    """Raised when a dataset is not production-ready for training."""


class DatasetLoadError(TrainingError):
    """Raised when dataset loading fails."""


class AugmentationError(TrainingError):
    """Raised when augmentation configuration or execution fails."""


class PreValidationError(TrainingError):
    """Raised when pre-training validation fails."""
