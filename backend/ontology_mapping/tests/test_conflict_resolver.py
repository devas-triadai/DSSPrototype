"""Tests for ConflictResolver."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.ontology_mapping.conflict_resolver import ConflictResolver
from backend.ontology_mapping.models import (
    ConflictType,
    MappingConflict,
    MappingRule,
    MatchType,
    ResolutionType,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver


@pytest.fixture
def resolver() -> ConflictResolver:
    OntologyResolver._tree = None
    return ConflictResolver()


@pytest.fixture
def duplicate_rules() -> list[MappingRule]:
    return [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
            confidence=1.0,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="car",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.EXACT,
            confidence=1.0,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture
def conflicting_rules() -> list[MappingRule]:
    return [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
            confidence=1.0,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="truck",
            canonical_value="ground_vehicle.heavy_equipment",
            match_type=MatchType.ALIAS,
            confidence=0.8,
            created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        ),
    ]


@pytest.fixture
def missing_node_rules() -> list[MappingRule]:
    return [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="dragon",
            canonical_value="mythical.dragon",
            match_type=MatchType.EXACT,
        ),
    ]


@pytest.mark.asyncio
async def test_detect_duplicates(
    resolver: ConflictResolver, duplicate_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(duplicate_rules)
    duplicate_conflicts = [
        c for c in conflicts if c.conflict_type == ConflictType.DUPLICATE
    ]
    assert len(duplicate_conflicts) >= 1


@pytest.mark.asyncio
async def test_detect_conflicting(
    resolver: ConflictResolver, conflicting_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(conflicting_rules)
    conflicting = [
        c for c in conflicts
        if c.conflict_type == ConflictType.CONFLICTING
    ]
    assert len(conflicting) >= 1


@pytest.mark.asyncio
async def test_detect_missing_node(
    resolver: ConflictResolver, missing_node_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(missing_node_rules)
    missing = [
        c for c in conflicts
        if c.conflict_type == ConflictType.MISSING_NODE
    ]
    assert len(missing) >= 1


@pytest.mark.asyncio
async def test_no_conflicts(resolver: ConflictResolver) -> None:
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
            source_label="truck",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.EXACT,
        ),
    ]
    conflicts = await resolver.detect(rules)
    conflicting = [
        c for c in conflicts
        if c.conflict_type == ConflictType.CONFLICTING
    ]
    assert len(conflicting) == 0


@pytest.mark.asyncio
async def test_resolve_first_wins(
    resolver: ConflictResolver, conflicting_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(conflicting_rules)
    if conflicts:
        resolution = await resolver.resolve(
            conflicts[0], ResolutionType.FIRST_WINS
        )
        assert resolution.resolution_type == ResolutionType.FIRST_WINS
        assert resolution.chosen_rule is not None
        assert resolution.chosen_rule.rule_id == "r1"


@pytest.mark.asyncio
async def test_resolve_last_wins(
    resolver: ConflictResolver, conflicting_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(conflicting_rules)
    if conflicts:
        resolution = await resolver.resolve(
            conflicts[0], ResolutionType.LAST_WINS
        )
        assert resolution.chosen_rule is not None
        assert resolution.chosen_rule.rule_id == "r2"


@pytest.mark.asyncio
async def test_resolve_highest_confidence(
    resolver: ConflictResolver, conflicting_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(conflicting_rules)
    if conflicts:
        resolution = await resolver.resolve(
            conflicts[0], ResolutionType.HIGHEST_CONFIDENCE
        )
        assert resolution.chosen_rule is not None
        assert resolution.chosen_rule.confidence == 1.0


@pytest.mark.asyncio
async def test_resolve_duplicate(
    resolver: ConflictResolver, duplicate_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(duplicate_rules)
    dup = [c for c in conflicts if c.conflict_type == ConflictType.DUPLICATE]
    if dup:
        resolution = await resolver.resolve(dup[0])
        assert resolution.chosen_rule is not None


@pytest.mark.asyncio
async def test_resolve_circular_needs_manual(
    resolver: ConflictResolver,
) -> None:
    conflict = MappingConflict(
        conflict_type=ConflictType.CIRCULAR,
        source_label="a",
        rules=(MappingRule(
            rule_id="circ",
            dataset_name="test",
            source_label="a",
            canonical_value="b",
            match_type=MatchType.ALIAS,
        ),),
        description="Circular alias chain",
    )
    resolution = await resolver.resolve(conflict)
    assert resolution.resolution_type == ResolutionType.MANUAL


@pytest.mark.asyncio
async def test_resolve_missing_node_needs_manual(
    resolver: ConflictResolver,
) -> None:
    conflict = MappingConflict(
        conflict_type=ConflictType.MISSING_NODE,
        source_label="dragon",
        rules=(
            MappingRule(
                rule_id="miss",
                dataset_name="test",
                source_label="dragon",
                canonical_value="mythical.dragon",
                match_type=MatchType.EXACT,
            ),
        ),
        description="Missing ontology node",
    )
    resolution = await resolver.resolve(conflict)
    assert resolution.resolution_type == ResolutionType.MANUAL


@pytest.mark.asyncio
async def test_resolve_all(
    resolver: ConflictResolver, conflicting_rules: list[MappingRule]
) -> None:
    conflicts = await resolver.detect(conflicting_rules)
    resolutions = await resolver.resolve_all(conflicts)
    assert len(resolutions) == len(conflicts)
    for r in resolutions:
        assert r.explanation is not None


@pytest.mark.asyncio
async def test_detect_ambiguous_aliases(resolver: ConflictResolver) -> None:
    rules = [
        MappingRule(
            rule_id="r1",
            dataset_name="coco",
            source_label="auto",
            canonical_value="ground_vehicle.car",
            match_type=MatchType.ALIAS,
        ),
        MappingRule(
            rule_id="r2",
            dataset_name="coco",
            source_label="auto",
            canonical_value="ground_vehicle.truck",
            match_type=MatchType.ALIAS,
        ),
    ]
    conflicts = await resolver.detect(rules)
    ambiguous = [
        c for c in conflicts
        if c.conflict_type == ConflictType.AMBIGUOUS_ALIAS
    ]
    assert len(ambiguous) >= 1


@pytest.mark.asyncio
async def test_resolve_empty_rules(resolver: ConflictResolver) -> None:
    conflict = MappingConflict(
        conflict_type=ConflictType.CONFLICTING,
        source_label="test",
        rules=(
            MappingRule(
                rule_id="empty-test",
                dataset_name="test",
                source_label="test",
                canonical_value="unknown_object",
                match_type=MatchType.SYNONYM,
            ),
        ),
        description="No rules",
    )
    resolution = await resolver.resolve(conflict)
    assert resolution.resolution_type is not None


@pytest.mark.asyncio
async def test_no_conflicts_for_distinct_labels(
    resolver: ConflictResolver,
) -> None:
    rules = [
        MappingRule(
            rule_id=str(i),
            dataset_name="coco",
            source_label=label,
            canonical_value=value,
            match_type=MatchType.EXACT,
        )
        for i, (label, value) in enumerate([
            ("car", "ground_vehicle.car"),
            ("truck", "ground_vehicle.truck"),
            ("person", "people.person"),
            ("bicycle", "ground_vehicle.bicycle"),
        ])
    ]
    conflicts = await resolver.detect(rules)
    conflicting = [
        c for c in conflicts
        if c.conflict_type in (
            ConflictType.DUPLICATE, ConflictType.CONFLICTING
        )
    ]
    assert len(conflicting) == 0
