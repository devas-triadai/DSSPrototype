"""Module-specific exceptions for the Dataset Intelligence Pipeline."""


class DatasetIntelligenceError(Exception):
    """Base exception for dataset intelligence errors."""


class ImportError(DatasetIntelligenceError):
    """Raised when dataset import fails."""


class FormatDetectionError(DatasetIntelligenceError):
    """Raised when format detection fails."""


class ValidationError(DatasetIntelligenceError):
    """Raised when dataset validation fails."""


class NormalizationError(DatasetIntelligenceError):
    """Raised when dataset normalization fails."""


class OntologyMappingError(DatasetIntelligenceError):
    """Raised when ontology mapping fails."""


class DuplicateDetectionError(DatasetIntelligenceError):
    """Raised when duplicate detection fails."""


class ClassHarmonizationError(DatasetIntelligenceError):
    """Raised when class harmonization fails."""


class MergeError(DatasetIntelligenceError):
    """Raised when dataset merging fails."""


class SplitError(DatasetIntelligenceError):
    """Raised when dataset splitting fails."""


class QualityAssessmentError(DatasetIntelligenceError):
    """Raised when quality assessment fails."""


class ExportError(DatasetIntelligenceError):
    """Raised when dataset export fails."""


class RegistryError(DatasetIntelligenceError):
    """Raised when registry operation fails."""


class DatasetNotFoundError(RegistryError):
    """Raised when a dataset is not found in the registry."""
