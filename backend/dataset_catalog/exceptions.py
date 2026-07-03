"""Module-specific exceptions for the Dataset Catalog System."""


class DatasetCatalogError(Exception):
    """Base exception for dataset catalog errors."""


class CatalogError(DatasetCatalogError):
    """Raised when a catalog operation fails."""


class SourceRegistryError(DatasetCatalogError):
    """Raised when a source registry operation fails."""


class ProfileError(DatasetCatalogError):
    """Raised when a profile operation fails."""


class TaxonomyError(DatasetCatalogError):
    """Raised when a taxonomy operation fails."""


class LicenseError(DatasetCatalogError):
    """Raised when a license operation fails."""


class CoverageError(DatasetCatalogError):
    """Raised when coverage analysis fails."""


class GapAnalysisError(DatasetCatalogError):
    """Raised when gap analysis fails."""


class AcquisitionError(DatasetCatalogError):
    """Raised when acquisition planning fails."""


class RecommendationError(DatasetCatalogError):
    """Raised when recommendation generation fails."""


class CurationError(DatasetCatalogError):
    """Raised when curation operations fail."""


class EntryNotFoundError(CatalogError):
    """Raised when a catalog entry is not found."""


class SourceNotFoundError(SourceRegistryError):
    """Raised when a source is not found."""


class TaxonomyNodeNotFoundError(TaxonomyError):
    """Raised when a taxonomy node is not found."""


class LicenseNotAllowedError(LicenseError):
    """Raised when a license is not in the allowed list."""


class InvalidScoreError(DatasetCatalogError):
    """Raised when a score is out of valid range."""


class CurationWorkflowError(CurationError):
    """Raised when a curation workflow fails."""


class AcquisitionLimitError(AcquisitionError):
    """Raised when acquisition limits are exceeded."""
