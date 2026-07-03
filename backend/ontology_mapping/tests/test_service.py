"""Tests for OntologyMappingService (public API)."""

from __future__ import annotations

import pytest

from backend.ontology_mapping.exceptions import DatasetNotFoundError
from backend.ontology_mapping.models import (
    DatasetProfile,
    MappingRule,
    MatchType,
    ResolutionType,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver
from backend.ontology_mapping.service import OntologyMappingService


@pytest.fixture
def service() -> OntologyMappingService:
    OntologyResolver._tree = None
    return OntologyMappingService()


@pytest.fixture
def coco_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="coco",
        version="2017",
        label_count=5,
        labels=("car", "truck", "person", "bus", "motorcycle"),
    )


@pytest.fixture
def coco_rules() -> list[MappingRule]:
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
        MappingRule(
            rule_id="r4",
            dataset_name="coco",
            source_label="bus",
            canonical_value="ground_vehicle.bus",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="r5",
            dataset_name="coco",
            source_label="motorcycle",
            canonical_value="ground_vehicle.motorcycle",
            match_type=MatchType.EXACT,
        ),
    ]


@pytest.mark.asyncio
async def test_register_dataset(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    mapping = await service.register_dataset(coco_profile, coco_rules)
    assert mapping.dataset_name == "coco"
    assert mapping.dataset_version == "2017"
    assert len(mapping.rules) == 5


@pytest.mark.asyncio
async def test_map_label(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    result = await service.map_label("coco", "car")
    assert result.canonical_value == "ground_vehicle.car"
    assert result.confidence == 1.0


@pytest.mark.asyncio
async def test_map_dataset(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    results = await service.map_dataset(
        "coco", ["car", "truck", "nonexistent_label_zzz"]
    )
    assert len(results) == 3
    assert results[0].canonical_value == "ground_vehicle.car"
    assert results[1].canonical_value == "ground_vehicle.truck"
    assert results[2].canonical_value == "unknown_object"


@pytest.mark.asyncio
async def test_validate_no_errors(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    errors = await service.validate("coco")
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_validate_nonexistent_dataset(
    service: OntologyMappingService,
) -> None:
    errors = await service.validate("nonexistent")
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_statistics(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    stats = await service.statistics("coco")
    assert stats.coverage_percent == 100.0
    assert stats.mapped_labels == 5


@pytest.mark.asyncio
async def test_statistics_nonexistent(
    service: OntologyMappingService,
) -> None:
    with pytest.raises(DatasetNotFoundError):
        await service.statistics("nonexistent")


@pytest.mark.asyncio
async def test_export_json(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    output = await service.export("coco", "json")
    assert "coco" in output
    assert "ground_vehicle.car" in output


@pytest.mark.asyncio
async def test_export_csv(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    output = await service.export("coco", "csv")
    assert "source_label" in output
    assert "car" in output


@pytest.mark.asyncio
async def test_export_nonexistent(
    service: OntologyMappingService,
) -> None:
    with pytest.raises(DatasetNotFoundError):
        await service.export("nonexistent", "json")


@pytest.mark.asyncio
async def test_detect_conflicts(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    conflicts = await service.detect_conflicts("coco")
    assert len(conflicts) == 0


@pytest.mark.asyncio
async def test_detect_conflicts_with_duplicates(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
) -> None:
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
            source_label="car",
            canonical_value="ground_vehicle.suv",
            match_type=MatchType.ALIAS,
        ),
    ]
    await service.register_dataset(coco_profile, rules)
    conflicts = await service.detect_conflicts("coco")
    assert len(conflicts) >= 1


@pytest.mark.asyncio
async def test_resolve_conflicts(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
) -> None:
    rules = [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
            confidence=1.0,
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.heavy_equipment",
            match_type=MatchType.ALIAS,
            confidence=0.7,
        ),
    ]
    await service.register_dataset(coco_profile, rules)
    resolutions = await service.resolve_conflicts(
        "coco", ResolutionType.HIGHEST_CONFIDENCE
    )
    assert len(resolutions) >= 1


@pytest.mark.asyncio
async def test_resolve(
    service: OntologyMappingService,
) -> None:
    res = await service.resolve("ground_vehicle.car")
    assert res.resolved_node is not None
    assert res.resolved_node.name == "Car"


@pytest.mark.asyncio
async def test_resolve_nonexistent(
    service: OntologyMappingService,
) -> None:
    res = await service.resolve("nonexistent")
    assert res.resolved_node is None


@pytest.mark.asyncio
async def test_get_datasets_empty(
    service: OntologyMappingService,
) -> None:
    datasets = await service.get_datasets()
    assert datasets == []


@pytest.mark.asyncio
async def test_get_datasets(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    datasets = await service.get_datasets()
    assert "coco" in datasets


@pytest.mark.asyncio
async def test_get_mapping(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    mapping = await service.get_mapping("coco")
    assert mapping is not None
    assert mapping.dataset_name == "coco"


@pytest.mark.asyncio
async def test_get_mapping_nonexistent(
    service: OntologyMappingService,
) -> None:
    mapping = await service.get_mapping("nonexistent")
    assert mapping is None


@pytest.mark.asyncio
async def test_remove_dataset(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    await service.remove_dataset("coco")
    datasets = await service.get_datasets()
    assert "coco" not in datasets


@pytest.mark.asyncio
async def test_register_alias_and_map(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    await service.register_alias("ground_vehicle.car", "automobile")
    result = await service.map_label("coco", "automobile")
    assert result.canonical_value == "ground_vehicle.car"


@pytest.mark.asyncio
async def test_register_synonym_and_map(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    await service.register_synonym("ground_vehicle.truck", "lorry")
    result = await service.map_label("coco", "lorry")
    assert result.canonical_value == "ground_vehicle.truck"


@pytest.mark.asyncio
async def test_register_regex_and_map(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    await service.register_regex(
        r"pick.?up", "ground_vehicle.pickup", 0.8
    )
    result = await service.map_label("coco", "pickup")
    assert result.canonical_value == "ground_vehicle.pickup"


@pytest.mark.asyncio
async def test_coverage_report(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    report = await service.coverage_report("coco")
    assert "HEALTHY" in report
    assert "100.0%" in report


@pytest.mark.asyncio
async def test_compatibility_report(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    profile2 = DatasetProfile(
        dataset_name="open_images",
        version="v6",
        label_count=1,
        labels=("car",),
    )
    rules2 = [
        MappingRule(
            rule_id="r1",
            dataset_name="open_images",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
    ]
    await service.register_dataset(profile2, rules2)
    report = await service.compatibility_report("coco", "open_images")
    assert "COMPATIBLE" in report


@pytest.mark.asyncio
async def test_validate_mapping_rules(
    service: OntologyMappingService,
) -> None:
    rules = [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
    ]
    errors = await service.validate_mapping_rules(rules)
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_validate_mapping_rules_invalid(
    service: OntologyMappingService,
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
    errors = await service.validate_mapping_rules(rules)
    assert len(errors) >= 1


@pytest.mark.asyncio
async def test_resolve_conflicts_no_conflicts(
    service: OntologyMappingService,
    coco_profile: DatasetProfile,
    coco_rules: list[MappingRule],
) -> None:
    await service.register_dataset(coco_profile, coco_rules)
    resolutions = await service.resolve_conflicts("coco")
    assert len(resolutions) == 0


@pytest.mark.asyncio
async def test_full_lifecycle(
    service: OntologyMappingService,
) -> None:
    profile = DatasetProfile(
        dataset_name="visdrone",
        version="2019",
        label_count=10,
        labels=("car", "truck", "pedestrian"),
    )
    rules = [
        MappingRule(
            rule_id="v1",
            dataset_name="visdrone",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="v2",
            dataset_name="visdrone",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        ),
        MappingRule(
            rule_id="v3",
            dataset_name="visdrone",
            source_label="pedestrian",
            canonical_value="people.person",
            match_type=MatchType.ALIAS,
        ),
    ]
    mapping = await service.register_dataset(profile, rules)
    assert len(mapping.rules) == 3

    result = await service.map_label("visdrone", "pedestrian")
    assert result.canonical_value == "people.person"

    results = await service.map_dataset(
        "visdrone", ["car", "nonexistent_xyz"]
    )
    assert results[1].canonical_value == "unknown_object"

    stats = await service.statistics("visdrone")
    assert stats.mapped_labels == 3

    errors = await service.validate("visdrone")
    assert len(errors) == 0

    json_out = await service.export("visdrone", "json")
    assert "visdrone" in json_out

    await service.remove_dataset("visdrone")
    assert "visdrone" not in await service.get_datasets()
