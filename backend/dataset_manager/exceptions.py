"""Module-specific exceptions for the Dataset Management Platform."""


class DatasetError(Exception):
    """Base exception for dataset-related errors."""


class DatasetNotFoundError(DatasetError):
    """Raised when a dataset cannot be found in the registry."""


class ValidationError(DatasetError):
    """Raised when dataset validation fails catastrophically."""


class SplitError(DatasetError):
    """Raised when dataset splitting fails."""


class ExportError(DatasetError):
    """Raised when dataset export fails."""


class ChecksumVerificationError(DatasetError):
    """Raised when checksum verification fails."""


class VersionError(DatasetError):
    """Raised when version creation or retrieval fails."""
