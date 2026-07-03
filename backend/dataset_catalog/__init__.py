"""Dataset Acquisition & Curation System.

Evaluates, scores, prioritizes, and curates candidate datasets before they
enter the Dataset Intelligence import pipeline.
"""

from backend.dataset_catalog.acquisition_planner import AcquisitionPlanner
from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.config import DatasetCatalogConfig, dc_config
from backend.dataset_catalog.coverage_analyzer import CoverageAnalyzer
from backend.dataset_catalog.curation_service import CurationService
from backend.dataset_catalog.dataset_profile import DatasetProfiler
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
from backend.dataset_catalog.gap_analyzer import GapAnalyzer
from backend.dataset_catalog.interfaces import (
    AcquisitionPlannerInterface,
    CatalogInterface,
    CoverageAnalyzerInterface,
    CurationServiceInterface,
    DatasetCatalogServiceInterface,
    DatasetProfileInterface,
    GapAnalyzerInterface,
    LicenseManagerInterface,
    RecommendationEngineInterface,
    SourceRegistryInterface,
    TaxonomicalCoverageInterface,
)
from backend.dataset_catalog.license_manager import LicenseManager
from backend.dataset_catalog.models import (
    AcquisitionPlan,
    CatalogEntry,
    ClassDistribution,
    CoverageReport,
    CurationRecord,
    DatasetProfile,
    GapAnalysisReport,
    GapEntry,
    LicenseInfo,
    RecommendationResult,
    SourceInfo,
    TaxonomicalCoverage,
    TaxonomyNode,
)
from backend.dataset_catalog.recommendation_engine import RecommendationEngine
from backend.dataset_catalog.service import DatasetCatalogService
from backend.dataset_catalog.source_registry import SourceRegistry

__all__ = [
    "DatasetCatalogConfig",
    "dc_config",
    "DatasetCatalogError",
    "CatalogError",
    "EntryNotFoundError",
    "SourceRegistryError",
    "SourceNotFoundError",
    "ProfileError",
    "TaxonomyError",
    "TaxonomyNodeNotFoundError",
    "LicenseError",
    "LicenseNotAllowedError",
    "CoverageError",
    "GapAnalysisError",
    "AcquisitionError",
    "AcquisitionLimitError",
    "RecommendationError",
    "CurationError",
    "CurationWorkflowError",
    "InvalidScoreError",
    "CatalogInterface",
    "SourceRegistryInterface",
    "DatasetProfileInterface",
    "TaxonomicalCoverageInterface",
    "LicenseManagerInterface",
    "AcquisitionPlannerInterface",
    "CoverageAnalyzerInterface",
    "GapAnalyzerInterface",
    "RecommendationEngineInterface",
    "CurationServiceInterface",
    "DatasetCatalogServiceInterface",
    "TaxonomyNode",
    "SourceInfo",
    "LicenseInfo",
    "ClassDistribution",
    "DatasetProfile",
    "CatalogEntry",
    "TaxonomicalCoverage",
    "CoverageReport",
    "GapEntry",
    "GapAnalysisReport",
    "AcquisitionPlan",
    "RecommendationResult",
    "CurationRecord",
    "Catalog",
    "SourceRegistry",
    "DatasetProfiler",
    "ClassTaxonomy",
    "LicenseManager",
    "AcquisitionPlanner",
    "CoverageAnalyzer",
    "GapAnalyzer",
    "RecommendationEngine",
    "CurationService",
    "DatasetCatalogService",
]
