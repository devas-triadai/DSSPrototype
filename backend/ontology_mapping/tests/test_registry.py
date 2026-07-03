"""Tests for MappingRegistry."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.exceptions import (
    DatasetAlreadyRegisteredError,
    DatasetNotFoundError,
)
from backend.ontology_mapping.models import MappingRule, MatchType
from backend.ontology_mapping.registry import MappingRegistry


@pytest.fixture
async def populated_registry() -> MappingRegistry:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    await reg.register_dataset("open_images")
    await reg.register_rule(
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        )
    )
    await reg.register_rule(
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        )
    )
    await reg.register_rule(
        MappingRule(
            rule_id="r3",
            dataset_name="open_images",
            source_label="Vehicle",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.ALIAS,
        )
    )
    return reg


@pytest.mark.asyncio
async def test_register_dataset() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    datasets = await reg.get_datasets()
    assert "coco" in datasets


@pytest.mark.asyncio
async def test_register_duplicate_dataset() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    with pytest.raises(DatasetAlreadyRegisteredError):
        await reg.register_dataset("coco")


@pytest.mark.asyncio
async def test_remove_dataset() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    await reg.remove_dataset("coco")
    assert "coco" not in await reg.get_datasets()


@pytest.mark.asyncio
async def test_remove_nonexistent_dataset() -> None:
    reg = MappingRegistry()
    with pytest.raises(DatasetNotFoundError):
        await reg.remove_dataset("nonexistent")


@pytest.mark.asyncio
async def test_register_and_lookup_rule() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    rule = MappingRule(
        rule_id="r1",
        dataset_name="coco",
        source_label="car",
        canonical_value="ground_vehicle.car",
        match_type=MatchType.EXACT,
    )
    await reg.register_rule(rule)
    results = await reg.lookup("coco", "car")
    assert len(results) == 1
    assert results[0].rule_id == "r1"


@pytest.mark.asyncio
async def test_lookup_case_insensitive() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    rule = MappingRule(
        rule_id="r1",
        dataset_name="coco",
        source_label="Car",
        canonical_value="ground_vehicle.car",
        match_type=MatchType.EXACT,
    )
    await reg.register_rule(rule)
    results = await reg.lookup("coco", "CAR")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_lookup_nonexistent_dataset() -> None:
    reg = MappingRegistry()
    with pytest.raises(DatasetNotFoundError):
        await reg.lookup("nonexistent", "car")


@pytest.mark.asyncio
async def test_get_rules(populated_registry: MappingRegistry) -> None:
    rules = await populated_registry.get_rules("coco")
    assert len(rules) == 2
    rule_ids = {r.rule_id for r in rules}
    assert rule_ids == {"r1", "r2"}


@pytest.mark.asyncio
async def test_get_rules_nonexistent() -> None:
    reg = MappingRegistry()
    with pytest.raises(DatasetNotFoundError):
        await reg.get_rules("nonexistent")


@pytest.mark.asyncio
async def test_remove_rule(populated_registry: MappingRegistry) -> None:
    await populated_registry.remove_rule("r1")
    rules = await populated_registry.get_rules("coco")
    assert len(rules) == 1
    assert rules[0].rule_id == "r2"


@pytest.mark.asyncio
async def test_remove_nonexistent_rule(
    populated_registry: MappingRegistry,
) -> None:
    await populated_registry.remove_rule("nonexistent")
    rules = await populated_registry.get_rules("coco")
    assert len(rules) == 2


@pytest.mark.asyncio
async def test_get_datasets(populated_registry: MappingRegistry) -> None:
    datasets = await populated_registry.get_datasets()
    assert sorted(datasets) == ["coco", "open_images"]


@pytest.mark.asyncio
async def test_version_before_commit() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    v = await reg.version("coco")
    assert v is None


@pytest.mark.asyncio
async def test_commit_version(populated_registry: MappingRegistry) -> None:
    v = await populated_registry.commit_version(
        "coco", "Initial version"
    )
    assert v.changelog == "Initial version"
    assert v.rule_count == 2
    assert v.dataset_count == 2


@pytest.mark.asyncio
async def test_history(populated_registry: MappingRegistry) -> None:
    await populated_registry.commit_version("coco", "v1")
    await populated_registry.commit_version("coco", "v2")
    history = await populated_registry.history("coco")
    assert len(history) == 2
    assert history[-1].changelog == "v2"


@pytest.mark.asyncio
async def test_clear() -> None:
    reg = MappingRegistry()
    await reg.register_dataset("coco")
    await reg.clear()
    assert await reg.get_datasets() == []


@pytest.mark.asyncio
async def test_ontology_version() -> None:
    reg = MappingRegistry()
    assert reg.ontology_version == "1.0.0"
    reg.ontology_version = "2.0.0"
    assert reg.ontology_version == "2.0.0"
