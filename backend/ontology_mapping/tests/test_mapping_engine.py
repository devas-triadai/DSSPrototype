"""Tests for MappingEngine."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.mapping_engine import MappingEngine
from backend.ontology_mapping.models import (
    MappingRule,
    MatchType,
)


@pytest.fixture
def engine() -> MappingEngine:
    return MappingEngine()


@pytest.fixture
def coco_rules() -> list[MappingRule]:
    return [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
        MappingRule(
            rule_id="r3",
            dataset_name="coco",
            source_label="person",
            canonical_value="people.person",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
        MappingRule(
            rule_id="r4",
            dataset_name="coco",
            source_label="motorcycle",
            canonical_value="ground_vehicle.motorcycle",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
        MappingRule(
            rule_id="r5",
            dataset_name="coco",
            source_label="bus",
            canonical_value="ground_vehicle.bus",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
    ]


@pytest.mark.asyncio
async def test_exact_match(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    result = await engine.map_label("coco", "car", coco_rules)
    assert result.canonical_value == "ground_vehicle.car"
    assert result.canonical_name == "Car"
    assert result.confidence == 1.0
    assert result.match_type == MatchType.EXACT


@pytest.mark.asyncio
async def test_case_insensitive_match(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    result = await engine.map_label("coco", "CAR", coco_rules)
    assert result.canonical_value == "ground_vehicle.car"
    assert result.confidence < 1.0


@pytest.mark.asyncio
async def test_no_match_returns_unknown(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    result = await engine.map_label("coco", "nonexistent", coco_rules)
    assert result.canonical_value == "unknown_object"
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_alias_match(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    engine.register_alias("ground_vehicle.car", "automobile")
    result = await engine.map_label("coco", "automobile", coco_rules)
    assert result.canonical_value == "ground_vehicle.car"


@pytest.mark.asyncio
async def test_synonym_match(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    engine.register_synonym("ground_vehicle.truck", "lorry")
    result = await engine.map_label("coco", "lorry", coco_rules)
    assert result.canonical_value == "ground_vehicle.truck"


@pytest.mark.asyncio
async def test_regex_match(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    engine.register_regex(r"pickup.*", "ground_vehicle.pickup", 0.8)
    result = await engine.map_label("coco", "pickup_truck", coco_rules)
    assert result.canonical_value == "ground_vehicle.pickup"


@pytest.mark.asyncio
async def test_plural_normalization(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    result = await engine.map_label("coco", "cars", coco_rules)
    assert result.canonical_value == "ground_vehicle.car"


@pytest.mark.asyncio
async def test_batch_mapping(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    results = await engine.map_batch(
        "coco",
        ["car", "truck", "person", "unmatched_xyz"],
        coco_rules,
    )
    assert len(results) == 4
    assert results[0].canonical_value == "ground_vehicle.car"
    assert results[1].canonical_value == "ground_vehicle.truck"
    assert results[2].canonical_value == "people.person"
    assert results[3].canonical_value == "unknown_object"


@pytest.mark.asyncio
async def test_alternatives_returned(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    engine.register_alias("ground_vehicle.car", "auto")
    engine.register_synonym("ground_vehicle.truck", "auto")
    result = await engine.map_label("coco", "auto", coco_rules)
    assert result.canonical_value is not None
    assert result.match_type is not None


@pytest.mark.asyncio
async def test_mapping_with_whitespace(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    result = await engine.map_label("coco", "  car  ", coco_rules)
    assert result.canonical_value == "ground_vehicle.car"


@pytest.mark.asyncio
async def test_mapping_unrecognized_label(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    result = await engine.map_label("coco", "zzznotfound", coco_rules)
    assert result.canonical_value == "unknown_object"


@pytest.mark.asyncio
async def test_multiple_aliases_same_value(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    engine.register_alias("ground_vehicle.car", "automobile")
    engine.register_alias("ground_vehicle.car", "auto")
    result = await engine.map_label("coco", "automobile", coco_rules)
    assert result.canonical_value == "ground_vehicle.car"


@pytest.mark.asyncio
async def test_register_regex_multiple(
    engine: MappingEngine, coco_rules: list[MappingRule]
) -> None:
    engine.register_regex(r"truck.*", "ground_vehicle.truck", 0.9)
    engine.register_regex(r"van.*", "ground_vehicle.van", 0.85)
    result = await engine.map_label("coco", "delivery_truck", coco_rules)
    assert result.canonical_value == "ground_vehicle.truck"
    assert result.confidence == 0.9
