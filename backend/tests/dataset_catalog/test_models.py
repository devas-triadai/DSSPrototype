"""Tests for Pydantic models."""


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


def test_taxonomy_node_defaults() -> None:
    node = TaxonomyNode(node_id="test_01", name="Test Node")
    assert node.node_id == "test_01"
    assert node.parent_id is None
    assert node.domain == "military"
    assert node.priority == 0.5
    assert node.child_ids == []


def test_taxonomy_node_with_children() -> None:
    node = TaxonomyNode(
        node_id="parent",
        name="Parent",
        parent_id=None,
        domain="military",
        priority=1.0,
        child_ids=["child_1", "child_2"],
    )
    assert node.child_ids == ["child_1", "child_2"]


def test_source_info_defaults() -> None:
    src = SourceInfo(source_id="src_001", name="Test Source", source_type="url")
    assert src.reliability == 0.5
    assert src.total_fetches == 0
    assert src.tags == []


def test_source_info_with_fetches() -> None:
    src = SourceInfo(
        source_id="src_002",
        name="Reliable Source",
        source_type="api",
        reliability=0.95,
        total_fetches=100,
        successful_fetches=95,
        failed_fetches=5,
    )
    assert src.reliability == 0.95
    assert src.total_fetches == 100


def test_license_info_defaults() -> None:
    lic = LicenseInfo(license_id="mit", name="MIT License")
    assert lic.spdx_identifier == ""
    assert lic.is_open is False
    assert lic.risk_score == 0.5


def test_license_info_open() -> None:
    lic = LicenseInfo(
        license_id="cc0",
        name="CC0",
        spdx_identifier="CC0-1.0",
        is_open=True,
        allows_commercial=True,
        risk_score=0.0,
    )
    assert lic.is_open is True
    assert lic.risk_score == 0.0


def test_class_distribution() -> None:
    cd = ClassDistribution(
        class_name="tank",
        count=50,
        annotation_count=120,
        image_count=30,
    )
    assert cd.class_name == "tank"
    assert cd.avg_width == 0.0


def test_dataset_profile_minimal() -> None:
    profile = DatasetProfile(
        profile_id="prof_001",
        source_id="src_001",
        source_type="local",
        path="/data/dataset",
    )
    assert profile.total_images == 0
    assert profile.classes == []
    assert profile.estimated_size_mb == 0.0


def test_dataset_profile_full() -> None:
    profile = DatasetProfile(
        profile_id="prof_002",
        source_id="src_002",
        source_type="url",
        path="/data/dataset2",
        total_images=100,
        total_annotations=500,
        total_classes=5,
        classes=["tank", "apc", "truck"],
        avg_width=640.0,
        avg_height=480.0,
        format_consistency=0.95,
    )
    assert profile.total_images == 100
    assert profile.total_classes == 5
    assert profile.format_consistency == 0.95


def test_catalog_entry_defaults() -> None:
    entry = CatalogEntry(
        entry_id="entry_001",
        name="Test Dataset",
        source_id="src_001",
        source_type="local",
    )
    assert entry.status == "discovered"
    assert entry.domain == "military"
    assert entry.overall_score == 0.0
    assert entry.estimated_budget == 0.0


def test_catalog_entry_with_scores() -> None:
    entry = CatalogEntry(
        entry_id="entry_002",
        name="Scored Dataset",
        source_id="src_002",
        source_type="api",
        quality_score=0.85,
        coverage_score=0.72,
        diversity_score=0.60,
        license_score=0.95,
        overall_score=0.80,
    )
    assert entry.overall_score == 0.80
    assert entry.quality_score == 0.85


def test_taxonomical_coverage() -> None:
    cov = TaxonomicalCoverage(
        total_nodes=10,
        covered_nodes=["n1", "n2"],
        uncovered_nodes=["n3", "n4"],
        partial_nodes=["n5"],
        coverage_ratio=0.2,
        domain_coverage={"military": 0.2},
    )
    assert cov.coverage_ratio == 0.2
    assert len(cov.covered_nodes) == 2


def test_coverage_report() -> None:
    report = CoverageReport(report_id="cr_001")
    assert report.total_entries_analyzed == 0
    assert report.aggregate_coverage == 0.0


def test_gap_entry() -> None:
    gap = GapEntry(
        node_id="n_001",
        node_name="Tanks",
        domain="military",
        severity=0.85,
        impact="critical",
    )
    assert gap.severity == 0.85
    assert gap.impact == "critical"


def test_gap_analysis_report() -> None:
    report = GapAnalysisReport(report_id="gap_001")
    assert report.gaps == []
    assert report.total_gap_count == 0


def test_acquisition_plan() -> None:
    plan = AcquisitionPlan(
        plan_id="plan_001",
        entry_ids=["entry_001", "entry_002"],
        priority=0.8,
    )
    assert plan.status == "draft"
    assert len(plan.entry_ids) == 2


def test_recommendation_result() -> None:
    rec = RecommendationResult(
        entry_id="entry_001",
        entry_name="Test",
        domain="military",
        quality_score=0.9,
        coverage_score=0.8,
        diversity_score=0.7,
        license_score=1.0,
        overall_score=0.85,
    )
    assert rec.overall_score == 0.85
    assert rec.entry_name == "Test"


def test_curation_record_defaults() -> None:
    rec = CurationRecord(
        record_id="cur_001",
        entry_id="entry_001",
        curator="analyst_01",
    )
    assert rec.status == "draft"
    assert rec.reviewer == ""
    assert rec.rejection_reason == ""
    assert rec.approved_at == ""
