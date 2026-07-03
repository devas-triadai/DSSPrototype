"""Abstract interfaces for every component in the Dataset Catalog System.

All concrete implementations depend on these contracts, never on each other.
The system is source-agnostic and format-agnostic.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

from backend.dataset_catalog.models import (
    AcquisitionPlan,
    CatalogEntry,
    CoverageReport,
    CurationRecord,
    DatasetProfile,
    GapAnalysisReport,
    LicenseInfo,
    RecommendationResult,
    SourceInfo,
    TaxonomicalCoverage,
    TaxonomyNode,
)


class CatalogInterface(ABC):
    """Contract for the persistent dataset catalog store."""

    @abstractmethod
    def add_entry(self, entry: CatalogEntry) -> CatalogEntry:
        """Add a new entry to the catalog."""

    @abstractmethod
    def get_entry(self, entry_id: str) -> CatalogEntry | None:
        """Retrieve a catalog entry by ID."""

    @abstractmethod
    def update_entry(self, entry: CatalogEntry) -> CatalogEntry:
        """Update an existing catalog entry."""

    @abstractmethod
    def remove_entry(self, entry_id: str) -> bool:
        """Remove a catalog entry by ID."""

    @abstractmethod
    def list_entries(
        self,
        status: str | None = None,
        source_type: str | None = None,
        domain: str | None = None,
    ) -> list[CatalogEntry]:
        """List catalog entries with optional filters."""

    @abstractmethod
    def search_entries(self, query: str) -> list[CatalogEntry]:
        """Search catalog entries by name, description, or keywords."""

    @abstractmethod
    def count_entries(self) -> int:
        """Return the total number of catalog entries."""


class SourceRegistryInterface(ABC):
    """Contract for tracking and managing dataset sources."""

    @abstractmethod
    def register_source(self, source: SourceInfo) -> SourceInfo:
        """Register a new dataset source."""

    @abstractmethod
    def get_source(self, source_id: str) -> SourceInfo | None:
        """Retrieve a source by ID."""

    @abstractmethod
    def update_source(self, source: SourceInfo) -> SourceInfo:
        """Update an existing source."""

    @abstractmethod
    def list_sources(self, source_type: str | None = None) -> list[SourceInfo]:
        """List all registered sources, optionally filtered by type."""

    @abstractmethod
    def record_success(self, source_id: str) -> None:
        """Record a successful fetch from a source."""

    @abstractmethod
    def record_failure(self, source_id: str, error: str) -> None:
        """Record a failed fetch from a source."""

    @abstractmethod
    def get_reliability(self, source_id: str) -> float:
        """Return the reliability score (0.0–1.0) for a source."""


class DatasetProfileInterface(ABC):
    """Contract for profiling candidate datasets."""

    @abstractmethod
    def profile(
        self,
        path: Path,
        source_id: str,
        source_type: str,
    ) -> DatasetProfile:
        """Profile a candidate dataset and return a structured profile.

        Inspects:
          - Image count, resolution distribution
          - Annotation count, class distribution
          - Format consistency
          - License metadata
          - Directory structure
        """

    @abstractmethod
    def update_profile(self, profile: DatasetProfile) -> DatasetProfile:
        """Update an existing profile with new information."""

    @abstractmethod
    def get_profile(self, profile_id: str) -> DatasetProfile | None:
        """Retrieve a profile by ID."""

    @abstractmethod
    def compare_profiles(
        self, profile_ids: Sequence[str]
    ) -> dict[str, object]:
        """Compare multiple profiles and return similarity/difference metrics."""


class TaxonomicalCoverageInterface(ABC):
    """Contract for analyzing coverage of the military ontology."""

    @abstractmethod
    def load_taxonomy(self, path: Path | None = None) -> list[TaxonomyNode]:
        """Load the taxonomy tree from disk or a source."""

    @abstractmethod
    def analyze_coverage(
        self,
        profile_classes: Sequence[str],
        taxonomy: Sequence[TaxonomyNode] | None = None,
    ) -> TaxonomicalCoverage:
        """Analyze how well profile classes cover the taxonomy."""

    @abstractmethod
    def get_coverage_report(
        self,
        profile: DatasetProfile,
        taxonomy: Sequence[TaxonomyNode] | None = None,
    ) -> CoverageReport:
        """Produce a full coverage report for a dataset profile."""


class LicenseManagerInterface(ABC):
    """Contract for managing dataset license information."""

    @abstractmethod
    def classify_license(self, license_text: str) -> LicenseInfo:
        """Classify a license string into a structured LicenseInfo."""

    @abstractmethod
    def is_allowed(self, license_info: LicenseInfo) -> bool:
        """Check if a license is in the allowed list."""

    @abstractmethod
    def is_restricted(self, license_info: LicenseInfo) -> bool:
        """Check if a license is in the restricted list."""

    @abstractmethod
    def compute_risk_score(self, license_info: LicenseInfo) -> float:
        """Compute a risk score (0.0–1.0) for a given license."""

    @abstractmethod
    def get_license_compatibility(
        self, license_a: LicenseInfo, license_b: LicenseInfo
    ) -> float:
        """Return compatibility score (0.0–1.0) between two licenses."""


class AcquisitionPlannerInterface(ABC):
    """Contract for planning dataset acquisitions."""

    @abstractmethod
    def create_plan(self, plan: AcquisitionPlan) -> AcquisitionPlan:
        """Create a new acquisition plan."""

    @abstractmethod
    def get_plan(self, plan_id: str) -> AcquisitionPlan | None:
        """Retrieve an acquisition plan by ID."""

    @abstractmethod
    def update_plan(self, plan: AcquisitionPlan) -> AcquisitionPlan:
        """Update an existing acquisition plan."""

    @abstractmethod
    def list_active_plans(self) -> list[AcquisitionPlan]:
        """List all currently active acquisition plans."""

    @abstractmethod
    def prioritize_plan(self, plan_id: str) -> AcquisitionPlan:
        """Recalculate priority for an acquisition plan."""


class CoverageAnalyzerInterface(ABC):
    """Contract for holistic coverage analysis across the catalog."""

    @abstractmethod
    def analyze_all(self) -> CoverageReport:
        """Analyze coverage across all catalog entries."""

    @abstractmethod
    def analyze_domain(self, domain: str) -> CoverageReport:
        """Analyze coverage for a specific domain."""

    @abstractmethod
    def analyze_entries(
        self, entry_ids: Sequence[str]
    ) -> CoverageReport:
        """Analyze coverage for a specific set of catalog entries."""

    @abstractmethod
    def coverage_trend(
        self, days: int = 30
    ) -> list[dict[str, object]]:
        """Return coverage metrics over time for trend analysis."""


class GapAnalyzerInterface(ABC):
    """Contract for identifying gaps in dataset coverage."""

    @abstractmethod
    def identify_gaps(
        self,
        coverage: CoverageReport,
        taxonomy: Sequence[TaxonomyNode] | None = None,
    ) -> GapAnalysisReport:
        """Identify gaps in coverage against the taxonomy."""

    @abstractmethod
    def get_critical_gaps(self, report: GapAnalysisReport) -> list[str]:
        """Return taxonomy node IDs that represent critical gaps."""

    @abstractmethod
    def get_gap_recommendations(
        self, report: GapAnalysisReport
    ) -> list[dict[str, object]]:
        """Return actionable recommendations for filling gaps."""


class RecommendationEngineInterface(ABC):
    """Contract for scoring and recommending datasets."""

    @abstractmethod
    def score_entry(self, entry_id: str) -> RecommendationResult:
        """Score a single catalog entry and return a recommendation."""

    @abstractmethod
    def recommend(
        self,
        domain: str | None = None,
        limit: int = 20,
        min_score: float = 0.3,
    ) -> list[RecommendationResult]:
        """Return the top-N dataset recommendations."""

    @abstractmethod
    def recommend_for_gap(
        self,
        gap_node_id: str,
        limit: int = 10,
    ) -> list[RecommendationResult]:
        """Recommend datasets that best fill a specific taxonomy gap."""


class CurationServiceInterface(ABC):
    """Contract for curation workflow management."""

    @abstractmethod
    def create_record(
        self, entry_id: str, curator: str
    ) -> CurationRecord:
        """Create a curation record for a catalog entry."""

    @abstractmethod
    def submit_for_review(self, record_id: str) -> CurationRecord:
        """Submit a curation record for review."""

    @abstractmethod
    def approve(self, record_id: str, reviewer: str) -> CurationRecord:
        """Approve a curation record."""

    @abstractmethod
    def reject(
        self, record_id: str, reviewer: str, reason: str
    ) -> CurationRecord:
        """Reject a curation record with a reason."""

    @abstractmethod
    def get_record(self, record_id: str) -> CurationRecord | None:
        """Retrieve a curation record by ID."""

    @abstractmethod
    def list_pending(self) -> list[CurationRecord]:
        """List all curation records pending review."""

    @abstractmethod
    def list_by_curator(self, curator: str) -> list[CurationRecord]:
        """List all curation records created by a curator."""


class DatasetCatalogServiceInterface(ABC):
    """Contract for the public Dataset Catalog service."""

    @abstractmethod
    def discover(
        self,
        path: Path,
        source_id: str,
        source_type: str,
        curator: str | None = None,
    ) -> CatalogEntry:
        """Discover and profile a candidate dataset, then add it to the catalog.

        This is the primary entry point for new datasets.
        """

    @abstractmethod
    def get_catalog_coverage(self) -> CoverageReport:
        """Get a holistic coverage report across the entire catalog."""

    @abstractmethod
    def get_gap_analysis(self) -> GapAnalysisReport:
        """Get a gap analysis report identifying coverage deficiencies."""

    @abstractmethod
    def get_recommendations(
        self,
        domain: str | None = None,
        limit: int = 20,
    ) -> list[RecommendationResult]:
        """Get the top dataset recommendations."""

    @abstractmethod
    def recommend_for_gap(
        self, gap_node_id: str
    ) -> list[RecommendationResult]:
        """Get recommendations to fill a specific taxonomy gap."""

    @abstractmethod
    def create_acquisition_plan(
        self,
        entries: Sequence[str],
        priority: float,
        notes: str = "",
    ) -> AcquisitionPlan:
        """Create an acquisition plan for one or more catalog entries."""

    @abstractmethod
    def submit_for_curation(
        self, entry_id: str, curator: str
    ) -> CurationRecord:
        """Submit a catalog entry for curation review."""

    @abstractmethod
    def approve_curation(
        self, record_id: str, reviewer: str
    ) -> CurationRecord:
        """Approve a pending curation record."""

    @abstractmethod
    def reject_curation(
        self, record_id: str, reviewer: str, reason: str
    ) -> CurationRecord:
        """Reject a pending curation record."""
