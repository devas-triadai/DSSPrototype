"""Strongly typed Pydantic models for the Dataset Catalog System.

Every model uses frozen=True for immutability, following the DSS contract pattern.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------


class TaxonomyNode(BaseModel):
    """A single node in the military domain taxonomy."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(..., description="Unique taxonomy node ID")
    name: str = Field(..., description="Human-readable name")
    parent_id: str | None = Field(default=None)
    domain: str = Field(default="military")
    description: str = Field(default="")
    keywords: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    child_ids: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """Information about a dataset source."""

    model_config = ConfigDict(frozen=True)

    source_id: str = Field(..., description="Unique source identifier")
    name: str = Field(...)
    source_type: str = Field(..., description="url | api | local | manual")
    url: str = Field(default="")
    description: str = Field(default="")
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    total_fetches: int = Field(default=0)
    successful_fetches: int = Field(default=0)
    failed_fetches: int = Field(default=0)
    last_fetch: str = Field(default="")
    last_error: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# License
# ---------------------------------------------------------------------------


class LicenseInfo(BaseModel):
    """Structured license information for a dataset."""

    model_config = ConfigDict(frozen=True)

    license_id: str = Field(...)
    name: str = Field(...)
    spdx_identifier: str = Field(default="")
    is_open: bool = Field(default=False)
    allows_commercial: bool = Field(default=False)
    allows_modification: bool = Field(default=False)
    requires_attribution: bool = Field(default=True)
    requires_share_alike: bool = Field(default=False)
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    compatible_licenses: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


class ClassDistribution(BaseModel):
    """Per-class statistics in a dataset profile."""

    model_config = ConfigDict(frozen=True)

    class_name: str = Field(...)
    count: int = Field(default=0)
    annotation_count: int = Field(default=0)
    image_count: int = Field(default=0)
    avg_width: float = Field(default=0.0)
    avg_height: float = Field(default=0.0)


class DatasetProfile(BaseModel):
    """A structured profile of a candidate dataset."""

    model_config = ConfigDict(frozen=True)

    profile_id: str = Field(...)
    source_id: str = Field(...)
    source_type: str = Field(...)
    path: str = Field(...)

    # Counts
    total_images: int = Field(default=0)
    total_annotations: int = Field(default=0)
    total_classes: int = Field(default=0)
    classes: list[str] = Field(default_factory=list)
    class_distribution: list[ClassDistribution] = Field(default_factory=list)

    # Resolution
    avg_width: float = Field(default=0.0)
    avg_height: float = Field(default=0.0)
    resolution_distribution: dict[str, int] = Field(default_factory=dict)

    # Quality indicators
    missing_annotations: int = Field(default=0)
    corrupt_images: int = Field(default=0)
    unsupported_formats: list[str] = Field(default_factory=list)
    annotation_format: str = Field(default="")
    format_consistency: float = Field(default=1.0, ge=0.0, le=1.0)

    # License
    license_info: LicenseInfo | None = Field(default=None)

    # Metadata
    estimated_size_mb: float = Field(default=0.0)
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    profiled_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class CatalogEntry(BaseModel):
    """A single entry in the dataset catalog."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(...)
    name: str = Field(...)
    source_id: str = Field(...)
    source_type: str = Field(...)
    domain: str = Field(default="military")
    status: str = Field(
        default="discovered",
        description="discovered | profiled | reviewed | acquired | rejected | archived",
    )
    profile: DatasetProfile | None = Field(default=None)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    diversity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    license_score: float = Field(default=1.0, ge=0.0, le=1.0)
    overall_score: float = Field(default=0.0, ge=0.0, le=1.0)
    estimated_budget: float = Field(default=0.0, description="Estimated acquisition budget")
    estimated_storage_mb: float = Field(default=0.0)
    tags: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


class TaxonomicalCoverage(BaseModel):
    """Coverage analysis of taxonomy nodes by a dataset."""

    model_config = ConfigDict(frozen=True)

    taxonomy_version: str = Field(default="")
    total_nodes: int = Field(default=0)
    covered_nodes: list[str] = Field(default_factory=list)
    uncovered_nodes: list[str] = Field(default_factory=list)
    partial_nodes: list[str] = Field(default_factory=list)
    coverage_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_coverage: dict[str, float] = Field(default_factory=dict)


class CoverageReport(BaseModel):
    """Holistic coverage report across catalog entries."""

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(...)
    total_entries_analyzed: int = Field(default=0)
    entries: list[TaxonomicalCoverage] = Field(default_factory=list)
    aggregate_coverage: float = Field(default=0.0, ge=0.0, le=1.0)
    domain_breakdown: dict[str, float] = Field(default_factory=dict)
    weakest_domains: list[str] = Field(default_factory=list)
    strongest_domains: list[str] = Field(default_factory=list)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Gap Analysis
# ---------------------------------------------------------------------------


class GapEntry(BaseModel):
    """A single identified gap in coverage."""

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(...)
    node_name: str = Field(...)
    domain: str = Field(...)
    severity: float = Field(..., ge=0.0, le=1.0)
    description: str = Field(default="")
    potential_sources: list[str] = Field(default_factory=list)
    impact: str = Field(default="medium", description="low | medium | high | critical")


class GapAnalysisReport(BaseModel):
    """Comprehensive gap analysis report."""

    model_config = ConfigDict(frozen=True)

    report_id: str = Field(...)
    gaps: list[GapEntry] = Field(default_factory=list)
    critical_gap_count: int = Field(default=0)
    high_gap_count: int = Field(default=0)
    medium_gap_count: int = Field(default=0)
    low_gap_count: int = Field(default=0)
    total_gap_count: int = Field(default=0)
    aggregate_gap_severity: float = Field(default=0.0)
    domain_breakdown: dict[str, int] = Field(default_factory=dict)
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


class AcquisitionPlan(BaseModel):
    """Plan for acquiring one or more datasets."""

    model_config = ConfigDict(frozen=True)

    plan_id: str = Field(...)
    entry_ids: list[str] = Field(default_factory=list)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    status: str = Field(
        default="draft",
        description="draft | active | in_progress | completed | cancelled",
    )
    estimated_budget: float = Field(default=0.0)
    estimated_storage_mb: float = Field(default=0.0)
    target_domains: list[str] = Field(default_factory=list)
    notes: str = Field(default="")
    created_by: str = Field(default="system")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------


class RecommendationResult(BaseModel):
    """Result of scoring a catalog entry."""

    model_config = ConfigDict(frozen=True)

    entry_id: str = Field(...)
    entry_name: str = Field(...)
    domain: str = Field(...)
    quality_score: float = Field(..., ge=0.0, le=1.0)
    coverage_score: float = Field(..., ge=0.0, le=1.0)
    diversity_score: float = Field(..., ge=0.0, le=1.0)
    license_score: float = Field(..., ge=0.0, le=1.0)
    overall_score: float = Field(..., ge=0.0, le=1.0)
    reason: str = Field(default="")
    recommended_for_gap: str = Field(default="")
    scored_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Curation
# ---------------------------------------------------------------------------


class CurationRecord(BaseModel):
    """Record of a curation workflow for a catalog entry."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(...)
    entry_id: str = Field(...)
    curator: str = Field(...)
    status: str = Field(default="draft", description="draft | pending_review | approved | rejected")
    reviewer: str = Field(default="")
    review_notes: str = Field(default="")
    rejection_reason: str = Field(default="")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    approved_at: str = Field(default="")
    rejected_at: str = Field(default="")
