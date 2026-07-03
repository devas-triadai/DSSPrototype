"""Tests for MappingStatisticsGenerator."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.mapping_statistics import (
    MappingStatisticsGenerator,
)
from backend.ontology_mapping.models import MappingRule, MatchType
from backend.ontology_mapping.ontology_resolver import OntologyResolver


@pytest.fixture
def stats() -> MappingStatisticsGenerator:
    OntologyResolver._tree = None
    return MappingStatisticsGenerator()


@pytest.fixture
def all_valid_rules() -> list[MappingRule]:
    return [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r3",
            dataset_name="coco",
            source_label="person",
            canonical_value="people.person",
            match_type=MatchType.EXACT,
        ),
    ]


@pytest.fixture
def mixed_rules() -> list[MappingRule]:
    return [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="dragon",
            canonical_value="mythical.dragon",
            match_type=MatchType.EXACT,
        ),
    ]


@pytest.mark.asyncio
async def test_coverage_100_percent(
    stats: MappingStatisticsGenerator, all_valid_rules: list[MappingRule]
) -> None:
    result = await stats.compute("coco", all_valid_rules)
    assert result.coverage_percent == 100.0
    assert result.mapped_labels == 3


@pytest.mark.asyncio
async def test_coverage_partial(
    stats: MappingStatisticsGenerator, mixed_rules: list[MappingRule]
) -> None:
    result = await stats.compute("coco", mixed_rules)
    assert result.coverage_percent == 50.0
    assert result.mapped_labels == 1
    assert result.unknown_labels == 1


@pytest.mark.asyncio
async def test_empty_rules(
    stats: MappingStatisticsGenerator,
) -> None:
    result = await stats.compute("coco", [])
    assert result.total_labels == 0
    assert result.coverage_percent == 0.0


@pytest.mark.asyncio
async def test_coverage_report_perfect(
    stats: MappingStatisticsGenerator, all_valid_rules: list[MappingRule]
) -> None:
    s = await stats.compute("coco", all_valid_rules)
    report = await stats.coverage_report(s)
    assert "HEALTHY" in report
    assert "100.0%" in report


@pytest.mark.asyncio
async def test_coverage_report_partial(
    stats: MappingStatisticsGenerator, mixed_rules: list[MappingRule]
) -> None:
    s = await stats.compute("coco", mixed_rules)
    report = await stats.coverage_report(s)
    assert "NEEDS_ATTENTION" in report


@pytest.mark.asyncio
async def test_coverage_report_acceptable() -> None:
    s = MappingStatisticsGenerator()
    stats_obj = await s.compute("test", [])
    stats_obj = stats_obj.model_copy(update={
        "mapped_labels": 8, "unknown_labels": 2, "total_labels": 10,
        "coverage_percent": 80.0,
    })
    report = await s.coverage_report(stats_obj)
    assert "ACCEPTABLE" in report


@pytest.mark.asyncio
async def test_compatibility_report_compatible() -> None:
    s = MappingStatisticsGenerator()
    a = await s.compute("a", [
        MappingRule(
            rule_id="r1", dataset_name="a",
            source_label="car", canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
    ])
    b = await s.compute("b", [
        MappingRule(
            rule_id="r1", dataset_name="b",
            source_label="automobile",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.ALIAS,
        ),
    ])
    report = await s.compatibility_report(a, b)
    assert "COMPATIBLE" in report


@pytest.mark.asyncio
async def test_duplicate_alias_count() -> None:
    s = MappingStatisticsGenerator()
    rules = [
        MappingRule(
            rule_id="r1", dataset_name="coco",
            source_label="car", canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r2", dataset_name="coco",
            source_label="car", canonical_value="ground_vehicle.suv",
            match_type=MatchType.ALIAS,
        ),
    ]
    result = await s.compute("coco", rules)
    assert result.duplicate_alias_count >= 1


@pytest.mark.asyncio
async def test_conflict_count() -> None:
    s = MappingStatisticsGenerator()
    rules = [
        MappingRule(
            rule_id="r1", dataset_name="coco",
            source_label="truck", canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r2", dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.heavy_equipment",
            match_type=MatchType.ALIAS,
        ),
    ]
    result = await s.compute("coco", rules)
    assert result.conflict_count >= 1


@pytest.mark.asyncio
async def test_statistics_generated_at() -> None:
    s = MappingStatisticsGenerator()
    result = await s.compute("test", [])
    assert result.generated_at is not None


@pytest.mark.asyncio
async def test_statistics_ontology_version() -> None:
    s = MappingStatisticsGenerator()
    result = await s.compute("test", [])
    assert result.ontology_version == "1.0.0"


@pytest.mark.asyncio
async def test_unknown_label_count(stats: MappingStatisticsGenerator) -> None:
    rules = [
        MappingRule(
            rule_id="r1", dataset_name="coco",
            source_label="car", canonical_value="unknown_object",
            match_type=MatchType.SYNONYM,
        ),
    ]
    result = await stats.compute("coco", rules)
    assert result.unknown_labels >= 1


@pytest.mark.asyncio
async def test_mapped_vs_unknown(
    stats: MappingStatisticsGenerator,
) -> None:
    rules = [
        MappingRule(
            rule_id="r1", dataset_name="coco",
            source_label="car", canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r2", dataset_name="coco",
            source_label="dragon",
            canonical_value="mythical.dragon",
            match_type=MatchType.EXACT,
        ),
    ]
    result = await stats.compute("coco", rules)
    assert result.mapped_labels + result.unknown_labels <= result.total_labels
