"""Tests for CoverageAnalyzer."""

import tempfile
from pathlib import Path

from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.coverage_analyzer import CoverageAnalyzer
from backend.dataset_catalog.models import CatalogEntry, DatasetProfile


def _setup_catalog(tmp: Path) -> tuple[Catalog, CoverageAnalyzer]:
    cat = Catalog(tmp / "catalog.json")
    taxonomy = ClassTaxonomy()
    analyzer = CoverageAnalyzer(cat, taxonomy, reports_dir=tmp / "reports")
    return cat, analyzer


def _add_entry(
    cat: Catalog, eid: str, classes: list[str], images: int = 10
) -> None:
    profile = DatasetProfile(
        profile_id=f"prof_{eid}",
        source_id="src_001",
        source_type="local",
        path=f"/data/{eid}",
        classes=classes,
        total_images=images,
    )
    entry = CatalogEntry(
        entry_id=eid,
        name=eid,
        source_id="src_001",
        source_type="local",
        profile=profile,
    )
    cat.add_entry(entry)


def test_analyze_all_empty() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, analyzer = _setup_catalog(Path(tmp))
        report = analyzer.analyze_all()
        assert report.total_entries_analyzed == 0
        assert report.aggregate_coverage == 0.0


def test_analyze_all_with_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat, analyzer = _setup_catalog(Path(tmp))
        _add_entry(cat, "e_001", ["tank", "apc", "ifv"])
        _add_entry(cat, "e_002", ["fighter_jet", "helicopter", "uav"])
        report = analyzer.analyze_all()
        assert report.total_entries_analyzed == 2
        assert report.aggregate_coverage > 0.0


def test_analyze_domain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat, analyzer = _setup_catalog(Path(tmp))
        _add_entry(cat, "e_001", ["tank", "apc"])
        _add_entry(cat, "e_002", ["fighter_jet"])
        report = analyzer.analyze_domain("military")
        assert report.total_entries_analyzed == 2


def test_analyze_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat, analyzer = _setup_catalog(Path(tmp))
        _add_entry(cat, "e_001", ["tank"])
        _add_entry(cat, "e_002", ["apc"])
        _add_entry(cat, "e_003", ["helicopter"])
        report = analyzer.analyze_entries(["e_001", "e_003"])
        assert report.total_entries_analyzed == 2


def test_coverage_trend() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, analyzer = _setup_catalog(Path(tmp))
        trend = analyzer.coverage_trend(7)
        assert len(trend) == 1


def test_analyze_all_with_unprofiled_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat, analyzer = _setup_catalog(Path(tmp))
        # Entry without profile
        entry = CatalogEntry(
            entry_id="e_no_profile",
            name="No Profile",
            source_id="src_001",
            source_type="local",
        )
        cat.add_entry(entry)
        _add_entry(cat, "e_profiled", ["tank"])
        report = analyzer.analyze_all()
        assert report.total_entries_analyzed == 1  # only profiled


def test_domain_breakdown() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat, analyzer = _setup_catalog(Path(tmp))
        _add_entry(cat, "e_001", ["tank", "apc", "ifv", "mrap", "mbt"])
        _add_entry(cat, "e_002", ["fighter_jet", "helicopter", "uav", "bomber"])
        _add_entry(cat, "e_003", ["aircraft_carrier", "destroyer", "submarine"])
        report = analyzer.analyze_all()
        assert "military" in report.domain_breakdown
        assert len(report.weakest_domains) > 0
        assert len(report.strongest_domains) > 0
