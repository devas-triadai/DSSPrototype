"""End-to-end tests for DatasetCatalogService."""

import tempfile
from pathlib import Path

from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.coverage_analyzer import CoverageAnalyzer
from backend.dataset_catalog.curation_service import CurationService
from backend.dataset_catalog.gap_analyzer import GapAnalyzer
from backend.dataset_catalog.license_manager import LicenseManager
from backend.dataset_catalog.models import CatalogEntry, DatasetProfile
from backend.dataset_catalog.recommendation_engine import RecommendationEngine
from backend.dataset_catalog.service import DatasetCatalogService
from backend.dataset_catalog.source_registry import SourceRegistry


def _setup_service(tmp: Path, with_entries: bool = False) -> DatasetCatalogService:
    catalog = Catalog(tmp / "catalog.json")
    sources = SourceRegistry(tmp / "sources.json")
    tax = ClassTaxonomy()
    lm = LicenseManager()
    coverage = CoverageAnalyzer(catalog, tax, reports_dir=tmp / "reports")
    gap = GapAnalyzer(tax, sources, reports_dir=tmp / "reports")
    rec = RecommendationEngine(catalog, tax, lm)
    curation = CurationService(catalog, work_dir=tmp / "work")

    svc = DatasetCatalogService(
        catalog=catalog,
        source_registry=sources,
        taxonomy=tax,
        license_manager=lm,
        coverage_analyzer=coverage,
        gap_analyzer=gap,
        recommendation_engine=rec,
        curation_service=curation,
    )

    if with_entries:
        profile = _make_profile("prof_a", ["tank", "apc", "ifv", "mbt", "mrap"])
        entry = CatalogEntry(
            entry_id="e_001",
            name="Ground Vehicles",
            source_id="src_001",
            source_type="local",
            status="profiled",
            profile=profile,
            quality_score=0.85,
            coverage_score=0.7,
            diversity_score=0.6,
            license_score=0.95,
            overall_score=0.78,
            estimated_budget=780.0,
            estimated_storage_mb=500.0,
        )
        catalog.add_entry(entry)

        profile2 = _make_profile("prof_b", ["fighter_jet", "helicopter", "uav"])
        entry2 = CatalogEntry(
            entry_id="e_002",
            name="Aircraft",
            source_id="src_002",
            source_type="api",
            status="profiled",
            profile=profile2,
            quality_score=0.9,
            coverage_score=0.6,
            diversity_score=0.5,
            license_score=1.0,
            overall_score=0.76,
            estimated_budget=760.0,
            estimated_storage_mb=300.0,
        )
        catalog.add_entry(entry2)

    return svc


def _make_profile(pid: str, classes: list[str]) -> DatasetProfile:
    from backend.dataset_catalog.models import ClassDistribution

    return DatasetProfile(
        profile_id=pid,
        source_id="src_001",
        source_type="local",
        path=f"/data/{pid}",
        total_images=100,
        total_annotations=200,
        total_classes=len(classes),
        classes=classes,
        class_distribution=[
            ClassDistribution(class_name=c, count=20) for c in classes
        ],
        format_consistency=0.95,
        estimated_size_mb=500.0,
    )


def test_discover(tmp_path: Path) -> None:
    svc = _setup_service(tmp_path)

    # Create a minimal dataset directory
    ds_dir = tmp_path / "candidate_ds"
    ds_dir.mkdir()
    (ds_dir / "img_001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    entry = svc.discover(
        path=ds_dir,
        source_id="src_001",
        source_type="local",
        curator=None,
    )
    assert entry.entry_id is not None
    assert entry.status == "profiled"
    assert entry.overall_score > 0.0
    assert entry.estimated_budget > 0.0


def test_discover_with_curator(tmp_path: Path) -> None:
    svc = _setup_service(tmp_path)
    ds_dir = tmp_path / "ds"
    ds_dir.mkdir()
    (ds_dir / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    entry = svc.discover(
        path=ds_dir,
        source_id="src_001",
        source_type="local",
        curator="analyst_01",
    )
    assert entry.status == "profiled"


def test_get_catalog_coverage_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp))
        report = svc.get_catalog_coverage()
        assert report.total_entries_analyzed == 0


def test_get_catalog_coverage_with_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)
        report = svc.get_catalog_coverage()
        assert report.total_entries_analyzed == 2
        assert report.aggregate_coverage > 0.0


def test_get_gap_analysis() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)
        report = svc.get_gap_analysis()
        assert report.total_gap_count > 0
        assert report.aggregate_gap_severity >= 0.0


def test_get_recommendations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)
        recs = svc.get_recommendations(limit=10)
        assert len(recs) == 2
        assert recs[0].overall_score >= recs[1].overall_score


def test_get_recommendations_with_domain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp))
        recs = svc.get_recommendations(domain="air", limit=10)
        assert len(recs) == 0


def test_recommend_for_gap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)
        results = svc.recommend_for_gap("tanks")
        assert len(results) >= 1


def test_create_acquisition_plan() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)
        plan = svc.create_acquisition_plan(
            entries=["e_001", "e_002"],
            priority=0.8,
            notes="High priority acquisition",
        )
        assert plan.plan_id is not None
        assert plan.priority == 0.8
        assert len(plan.entry_ids) == 2


def test_submit_and_approve_curation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)

        record = svc.submit_for_curation("e_001", "analyst_01")
        assert record.status == "pending_review"

        approved = svc.approve_curation(record.record_id, "manager_01")
        assert approved.status == "approved"

        e = svc._catalog.get_entry("e_001")
        assert e is not None
        assert e.status == "acquired"


def test_submit_and_reject_curation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        svc = _setup_service(Path(tmp), with_entries=True)

        record = svc.submit_for_curation("e_001", "analyst_01")
        rejected = svc.reject_curation(
            record.record_id, "manager_01", "Dataset does not meet requirements"
        )
        assert rejected.status == "rejected"
        assert rejected.rejection_reason == "Dataset does not meet requirements"


def test_default_constructor() -> None:
    svc = DatasetCatalogService()
    assert svc is not None
