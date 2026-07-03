"""Tests for ClassTaxonomy."""

import json
from pathlib import Path

from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.exceptions import TaxonomyNodeNotFoundError
from backend.dataset_catalog.models import DatasetProfile


def test_default_taxonomy_loaded() -> None:
    tax = ClassTaxonomy()
    assert tax.load_taxonomy() is not None
    assert len(tax.load_taxonomy()) > 10


def test_get_node() -> None:
    tax = ClassTaxonomy()
    node = tax.get_node("tanks")
    assert node.name == "Tanks"
    assert node.domain == "military"


def test_get_nonexistent_node_raises() -> None:
    tax = ClassTaxonomy()
    try:
        tax.get_node("nonexistent_node")
        assert False, "Expected TaxonomyNodeNotFoundError"
    except TaxonomyNodeNotFoundError:
        pass


def test_get_children() -> None:
    tax = ClassTaxonomy()
    children = tax.get_children("ground_vehicles")
    assert len(children) > 0
    assert any(c.node_id == "tanks" for c in children)


def test_get_descendants() -> None:
    tax = ClassTaxonomy()
    descendants = tax.get_descendants("ground_vehicles")
    assert len(descendants) > 3
    assert any(d.node_id == "mbt" for d in descendants)


def test_get_ancestors() -> None:
    tax = ClassTaxonomy()
    ancestors = tax.get_ancestors("mbt")
    ancestor_ids = [a.node_id for a in ancestors]
    assert "tanks" in ancestor_ids
    assert "ground_vehicles" in ancestor_ids


def test_find_nodes_by_keyword() -> None:
    tax = ClassTaxonomy()
    matches = tax.find_nodes_by_keyword("tank")
    assert len(matches) > 0
    assert any(m.node_id == "tanks" for m in matches)


def test_find_nodes_by_alias() -> None:
    tax = ClassTaxonomy()
    matches = tax.find_nodes_by_keyword("main battle tank")
    assert len(matches) > 0


def test_analyze_coverage_full() -> None:
    tax = ClassTaxonomy()
    classes = ["tank", "apc", "ifv", "fighter_jet", "helicopter", "uav",
               "mbt", "bmp", "bradley", "mrap"]
    coverage = tax.analyze_coverage(classes)
    assert coverage.coverage_ratio > 0.0
    assert len(coverage.covered_nodes) > 0


def test_analyze_coverage_empty() -> None:
    tax = ClassTaxonomy()
    coverage = tax.analyze_coverage([])
    assert coverage.coverage_ratio == 0.0
    assert len(coverage.covered_nodes) == 0


def test_analyze_coverage_partial() -> None:
    tax = ClassTaxonomy()
    coverage = tax.analyze_coverage(["tank", "helicopter"])
    assert 0.0 < coverage.coverage_ratio < 1.0
    domain = coverage.domain_coverage
    assert "military" in domain


def test_get_coverage_report() -> None:
    tax = ClassTaxonomy()
    profile = DatasetProfile(
        profile_id="prof_001",
        source_id="src_001",
        source_type="local",
        path="/data",
        classes=["tank", "apc", "fighter_jet", "helicopter"],
    )
    report = tax.get_coverage_report(profile)
    assert report.total_entries_analyzed == 1
    assert report.aggregate_coverage > 0.0
    assert "military" in report.domain_breakdown


def test_load_taxonomy_from_json(tmp_path: Path) -> None:
    tax_file = tmp_path / "taxonomy.json"
    data = {
        "version": "2.0.0",
        "nodes": [
            {
                "node_id": "test_node",
                "name": "Test Node",
                "parent_id": None,
                "domain": "test",
                "priority": 1.0,
            }
        ],
    }
    with tax_file.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    tax = ClassTaxonomy(taxonomy_path=tax_file)
    node = tax.get_node("test_node")
    assert node.name == "Test Node"
    assert tax.load_taxonomy() is not None


def test_get_coverage_report_empty_profile() -> None:
    tax = ClassTaxonomy()
    profile = DatasetProfile(
        profile_id="prof_empty",
        source_id="src_001",
        source_type="local",
        path="/empty",
    )
    report = tax.get_coverage_report(profile)
    assert report.aggregate_coverage == 0.0
