"""Tests for RecommendationEngine."""

from pathlib import Path

from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.class_taxonomy import ClassTaxonomy
from backend.dataset_catalog.license_manager import LicenseManager
from backend.dataset_catalog.models import (
    CatalogEntry,
    ClassDistribution,
    DatasetProfile,
)
from backend.dataset_catalog.recommendation_engine import RecommendationEngine


def _setup(tmp: Path) -> tuple[Catalog, RecommendationEngine]:
    cat = Catalog(tmp / "catalog.json")
    tax = ClassTaxonomy()
    lm = LicenseManager()
    engine = RecommendationEngine(cat, tax, lm)
    return cat, engine


def _add_entry(
    cat: Catalog,
    eid: str,
    name: str,
    classes: list[str] | None = None,
    counts: list[int] | None = None,
    domain: str = "military",
) -> None:
    classes = classes or ["tank"]
    cls_dists = [
        ClassDistribution(class_name=cls, count=counts[i] if counts else 10)
        for i, cls in enumerate(classes)
    ]
    profile = DatasetProfile(
        profile_id=f"prof_{eid}",
        source_id="src_001",
        source_type="local",
        path=f"/data/{eid}",
        total_images=100,
        total_annotations=200,
        total_classes=len(classes),
        classes=classes,
        class_distribution=cls_dists,
        format_consistency=0.95,
    )
    entry = CatalogEntry(
        entry_id=eid,
        name=name,
        source_id="src_001",
        source_type="local",
        domain=domain,
        profile=profile,
        quality_score=0.8,
        coverage_score=0.7,
        diversity_score=0.6,
        license_score=0.9,
        overall_score=0.75,
    )
    cat.add_entry(entry)


def test_score_entry(tmp_path: Path) -> None:
    cat, engine = _setup(tmp_path)
    _add_entry(cat, "e_001", "Test", classes=["tank", "apc", "ifv"])
    result = engine.score_entry("e_001")
    assert result.entry_id == "e_001"
    assert result.overall_score > 0.0
    assert result.quality_score >= 0.0


def test_recommend(tmp_path: Path) -> None:
    cat, engine = _setup(tmp_path)
    _add_entry(cat, "e_001", "Dataset A", classes=["tank"], counts=[50])
    _add_entry(cat, "e_002", "Dataset B", classes=["helicopter"], counts=[30])
    results = engine.recommend(limit=10, min_score=0.3)
    assert len(results) == 2
    assert results[0].overall_score >= 0.3


def test_recommend_with_domain_filter(tmp_path: Path) -> None:
    cat, engine = _setup(tmp_path)
    _add_entry(cat, "e_001", "Military", classes=["tank"], domain="military")
    _add_entry(cat, "e_002", "Civilian", classes=["car"], domain="civilian")
    results = engine.recommend(domain="military", limit=10, min_score=0.0)
    assert len(results) == 1
    assert results[0].entry_id == "e_001"


def test_recommend_for_gap(tmp_path: Path) -> None:
    cat, engine = _setup(tmp_path)
    _add_entry(cat, "e_001", "Tanks", classes=["tank", "mbt", "light_tank"])
    _add_entry(cat, "e_002", "Aircraft", classes=["fighter_jet", "helicopter"])
    results = engine.recommend_for_gap("tanks", limit=10)
    assert len(results) >= 1
    assert any(r.entry_id == "e_001" for r in results)


def test_recommend_min_score_filter(tmp_path: Path) -> None:
    cat, engine = _setup(tmp_path)
    _add_entry(cat, "e_001", "Good", classes=["tank"], counts=[50])
    _add_entry(cat, "e_002", "Poor", classes=[], counts=[])
    results = engine.recommend(limit=10, min_score=0.9)
    assert len(results) < 2


def test_recommendation_reason(tmp_path: Path) -> None:
    cat, engine = _setup(tmp_path)
    _add_entry(cat, "e_001", "High Quality", classes=["tank", "apc"], counts=[100, 80])
    result = engine.score_entry("e_001")
    assert len(result.reason) > 0
