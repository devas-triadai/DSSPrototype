"""Tests for Dataset Catalog exception hierarchy."""

from backend.dataset_catalog.exceptions import (
    AcquisitionError,
    AcquisitionLimitError,
    CatalogError,
    CoverageError,
    CurationError,
    CurationWorkflowError,
    DatasetCatalogError,
    EntryNotFoundError,
    GapAnalysisError,
    InvalidScoreError,
    LicenseError,
    LicenseNotAllowedError,
    ProfileError,
    RecommendationError,
    SourceNotFoundError,
    SourceRegistryError,
    TaxonomyError,
    TaxonomyNodeNotFoundError,
)


def test_base_exception() -> None:
    assert issubclass(CatalogError, DatasetCatalogError)
    assert issubclass(SourceRegistryError, DatasetCatalogError)
    assert issubclass(ProfileError, DatasetCatalogError)


def test_specific_exceptions_inherit_correctly() -> None:
    assert issubclass(EntryNotFoundError, CatalogError)
    assert issubclass(SourceNotFoundError, SourceRegistryError)
    assert issubclass(TaxonomyNodeNotFoundError, TaxonomyError)
    assert issubclass(LicenseNotAllowedError, LicenseError)
    assert issubclass(CurationWorkflowError, CurationError)
    assert issubclass(AcquisitionLimitError, AcquisitionError)


def test_exception_messages() -> None:
    e1 = EntryNotFoundError("Test entry not found")
    assert str(e1) == "Test entry not found"

    e2 = CurationWorkflowError("Invalid workflow state")
    assert str(e2) == "Invalid workflow state"

    e3 = AcquisitionLimitError("Too many active plans")
    assert str(e3) == "Too many active plans"

    e4 = TaxonomyNodeNotFoundError("Node not found: n_001")
    assert str(e4) == "Node not found: n_001"


def test_all_exceptions_under_base() -> None:
    all_exceptions = [
        CatalogError,
        SourceRegistryError,
        ProfileError,
        TaxonomyError,
        LicenseError,
        CoverageError,
        GapAnalysisError,
        AcquisitionError,
        RecommendationError,
        CurationError,
        InvalidScoreError,
    ]
    for exc in all_exceptions:
        assert issubclass(exc, DatasetCatalogError)
