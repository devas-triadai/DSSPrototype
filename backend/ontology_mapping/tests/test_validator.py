"""Tests for MappingValidator."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.mapping_validator import MappingValidator
from backend.ontology_mapping.models import MappingRule, MatchType
from backend.ontology_mapping.ontology_resolver import OntologyResolver


@pytest.fixture
def validator() -> MappingValidator:
    OntologyResolver._tree = None
    return MappingValidator()


@pytest.mark.asyncio
async def test_validate_ontology_no_errors(
    validator: MappingValidator,
) -> None:
    errors = await validator.validate_ontology()
    assert len(errors) == 0, f"Ontology validation errors: {errors}"


@pytest.mark.asyncio
async def test_validate_valid_mapping(validator: MappingValidator) -> None:
    rules = [
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
            source_label="person",
            canonical_value="people.person",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await validator.validate_mapping(rules)
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_validate_missing_node(
    validator: MappingValidator,
) -> None:
    rules = [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="dragon",
            canonical_value="mythical.dragon",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await validator.validate_mapping(rules)
    assert len(errors) >= 1
    assert "mythical.dragon" in errors[0]


@pytest.mark.asyncio
async def test_validate_conflicting_mapping(
    validator: MappingValidator,
) -> None:
    rules = [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.heavy_equipment",
            match_type=MatchType.ALIAS,
        ),
    ]
    errors = await validator.validate_mapping(rules)
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_validate_rules_empty_source(
    validator: MappingValidator,
) -> None:
    rules = [
        MappingRule.model_construct(
            rule_id="r1",
            dataset_name="coco",
            source_label="",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await validator.validate_rules(rules)
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_validate_rules_empty_canonical(
    validator: MappingValidator,
) -> None:
    rules = [
        MappingRule.model_construct(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await validator.validate_rules(rules)
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_validate_rules_invalid_confidence(
    validator: MappingValidator,
) -> None:
    rules = [
        MappingRule.model_construct(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
            confidence=1.5,
        ),
    ]
    errors = await validator.validate_rules(rules)
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_validate_rules_duplicate_id(
    validator: MappingValidator,
) -> None:
    rules = [
        MappingRule(
            rule_id="same-id",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="same-id",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await validator.validate_rules(rules)
    assert len(errors) >= 1
    assert "same-id" in errors[0]


@pytest.mark.asyncio
async def test_validate_rules_empty_id(validator: MappingValidator) -> None:
    rules = [
        MappingRule(
            rule_id="",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await validator.validate_rules(rules)
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_validate_rules_clean(validator: MappingValidator) -> None:
    rules = [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
    ]
    errors = await validator.validate_rules(rules)
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_validate_ontology_structure(
    validator: MappingValidator,
) -> None:
    errors = await validator.validate_ontology()
    assert len(errors) == 0
