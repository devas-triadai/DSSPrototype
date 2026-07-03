"""Tests for GapAnalyzer."""

from datetime import datetime, timezone

from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.gap_analyzer import GapAnalyzer
from backend.dataset_catalog.models import CoverageReport, TaxonomicalCoverage


def _make_coverage(
    covered: list[str] | None = None,
    partial: list[str] | None = None,
) -> CoverageReport:
    tax = ClassTaxonomy()
    all_nodes = tax.load_taxonomy()
    entries = [
        TaxonomicalCoverage(
            total_nodes=len(all_nodes),
            covered_nodes=covered or [],
            uncovered_nodes=[],
            partial_nodes=partial or [],
            coverage_ratio=(
                len(covered or []) / len(all_nodes) if all_nodes else 0.0
            ),
            domain_coverage={"military": 0.0},
        )
    ]
    return CoverageReport(
        report_id=f"cr_{datetime.now(timezone.utc).timestamp()}",
        total_entries_analyzed=1,
        entries=entries,
        aggregate_coverage=entries[0].coverage_ratio,
        domain_breakdown={"military": entries[0].coverage_ratio},
    )


def test_identify_gaps_all_uncovered() -> None:
    analyzer = GapAnalyzer(ClassTaxonomy())
    coverage = _make_coverage(covered=[])
    report = analyzer.identify_gaps(coverage)
    assert report.total_gap_count > 0
    assert report.critical_gap_count > 0


def test_identify_gaps_partial_coverage() -> None:
    tax = ClassTaxonomy()
    analyzer = GapAnalyzer(tax)
    top_level = ["ground_vehicles", "aircraft", "naval_vessels"]
    coverage = _make_coverage(covered=top_level)
    report = analyzer.identify_gaps(coverage)
    assert report.total_gap_count > 0


def test_identify_gaps_full_coverage() -> None:
    tax = ClassTaxonomy()
    analyzer = GapAnalyzer(tax)
    all_nodes = [n.node_id for n in tax.load_taxonomy()]
    coverage = _make_coverage(covered=all_nodes)
    report = analyzer.identify_gaps(coverage)
    assert report.total_gap_count == 0


def test_get_critical_gaps() -> None:
    analyzer = GapAnalyzer(ClassTaxonomy())
    coverage = _make_coverage(covered=[])
    report = analyzer.identify_gaps(coverage)
    critical = analyzer.get_critical_gaps(report)
    assert len(critical) > 0


def test_get_gap_recommendations() -> None:
    analyzer = GapAnalyzer(ClassTaxonomy())
    coverage = _make_coverage(covered=[])
    report = analyzer.identify_gaps(coverage)
    recs = analyzer.get_gap_recommendations(report)
    assert len(recs) > 0
    for rec in recs:
        assert "node_id" in rec
        assert "priority" in rec
        assert "action" in rec


def test_gap_severity_reflects_priority() -> None:
    tax = ClassTaxonomy()
    analyzer = GapAnalyzer(tax)
    coverage = _make_coverage(covered=[])
    report = analyzer.identify_gaps(coverage)
    # High-priority nodes should have higher severity
    tank_gaps = [g for g in report.gaps if g.node_id == "tanks"]
    if tank_gaps:
        assert tank_gaps[0].severity >= 0.5
