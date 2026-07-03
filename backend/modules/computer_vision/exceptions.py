"""Module-specific exceptions for the Computer Vision pipeline."""


class ModelLoadError(RuntimeError):
    """Raised when a vision model cannot be loaded or initialised."""


class InferenceError(RuntimeError):
    """Raised when model inference fails at runtime."""


class ImageValidationError(ValueError):
    """Raised when an image fails validation checks."""


class PreprocessingError(RuntimeError):
    """Raised when image preprocessing fails."""
