"""Test configuration and shared fixtures for ontology mapping tests."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from backend.ontology_mapping.models import (
    ConflictType,
    DatasetProfile,
    MappingConflict,
    MappingRule,
    MatchType,
    OntologyNode,
)


@pytest.fixture
def sample_rule_car() -> MappingRule:
    return MappingRule(
        rule_id="test-rule-car",
        dataset_name="coco",
        source_label="car",
        canonical_value="ground_vehicle.car",
        match_type=MatchType.EXACT,
        confidence=1.0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        version="1.0.0",
    )


@pytest.fixture
def sample_rule_truck() -> MappingRule:
    return MappingRule(
        rule_id="test-rule-truck",
        dataset_name="coco",
        source_label="truck",
        canonical_value="ground_vehicle.truck",
        match_type=MatchType.EXACT,
        confidence=1.0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        version="1.0.0",
    )


@pytest.fixture
def sample_rule_person() -> MappingRule:
    return MappingRule(
        rule_id="test-rule-person",
        dataset_name="coco",
        source_label="person",
        canonical_value="people.person",
        match_type=MatchType.EXACT,
        confidence=1.0,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        version="1.0.0",
    )


@pytest.fixture
def sample_profile() -> DatasetProfile:
    return DatasetProfile(
        dataset_name="coco",
        version="2017",
        label_count=80,
        labels=("car", "truck", "person", "bus", "bicycle"),
    )


@pytest.fixture
def sample_conflict() -> MappingConflict:
    return MappingConflict(
        conflict_id=str(uuid4()),
        conflict_type=ConflictType.CONFLICTING,
        source_label="truck",
        rules=(
            MappingRule(
                rule_id="rule-a",
                dataset_name="coco",
                source_label="truck",
                canonical_value="ground_vehicle.truck",
                match_type=MatchType.EXACT,
                confidence=1.0,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                version="1.0.0",
            ),
            MappingRule(
                rule_id="rule-b",
                dataset_name="coco",
                source_label="truck",
                canonical_value="ground_vehicle.heavy_equipment",
                match_type=MatchType.SYNONYM,
                confidence=0.7,
                created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
                version="1.0.0",
            ),
        ),
        description="Truck maps to two different ontology values",
    )


@pytest.fixture
def sample_ontology_node() -> OntologyNode:
    return OntologyNode(
        value="ground_vehicle.car",
        name="Car",
        category="ground_vehicle",
        category_name="Ground Vehicle",
        parent="ground_vehicle",
        children=frozenset(),
        depth=2,
        is_leaf=True,
        enum_member_name="GROUND_VEHICLE_CAR",
    )
