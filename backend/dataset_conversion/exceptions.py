from __future__ import annotations


class ConversionError(RuntimeError):
    """Base exception for all dataset conversion errors."""


class LoadError(ConversionError):
    """Raised when a dataset fails to load."""


class AnnotationError(ConversionError):
    """Raised when annotation parsing fails."""


class OntologyAdapterError(ConversionError):
    """Raised when ontology label translation fails."""


class GeometryError(ConversionError):
    """Raised when geometry conversion fails."""


class ImageError(ConversionError):
    """Raised when image validation or conversion fails."""


class MergeError(ConversionError):
    """Raised when dataset merging fails."""


class SplitError(ConversionError):
    """Raised when dataset splitting fails."""


class ValidationError(ConversionError):
    """Raised when dataset validation fails."""


class StatisticsError(ConversionError):
    """Raised when statistics computation fails."""


class ManifestError(ConversionError):
    """Raised when manifest generation fails."""


class ExportError(ConversionError):
    """Raised when dataset export fails."""


class PipelineError(ConversionError):
    """Raised when the conversion pipeline encounters an error."""
