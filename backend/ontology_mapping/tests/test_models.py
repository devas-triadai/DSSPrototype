"""Tests for all Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.ontology_mapping.models import (
    ConflictResolution,
    ConflictType,
    DatasetLabel,
    DatasetMapping,
    DatasetProfile,
    ExportFormat,
    MappingConflict,
    MappingResult,
    MappingRule,
    MappingStatistics,
    MappingVersion,
    MatchType,
    OntologyNode,
    OntologyResolution,
    ResolutionType,
)


def test_ontology_node_minimal() -> None:
    node = OntologyNode(
        value="ground_vehicle.car",
        name="Car",
        category="ground_vehicle",
        category_name="Ground Vehicle",
        parent="ground_vehicle",
        depth=2,
        is_leaf=True,
    )
    assert node.value == "ground_vehicle.car"
    assert node.name == "Car"
    assert node.depth == 2
    assert node.is_leaf
    assert node.children == frozenset()


def test_ontology_node_with_children() -> None:
    node = OntologyNode(
        value="ground_vehicle",
        name="Ground Vehicle",
        category="ground_vehicle",
        category_name="Ground Vehicle",
        parent="root",
        children=frozenset({"ground_vehicle.car", "ground_vehicle.truck"}),
        depth=1,
        is_leaf=False,
        enum_member_name=None,
    )
    assert "ground_vehicle.car" in node.children
    assert not node.is_leaf
    assert node.enum_member_name is None


def test_dataset_label_defaults() -> None:
    label = DatasetLabel(
        dataset_name="coco",
        original_label="Truck",
    )
    assert label.normalized_label is None
    assert label.confidence == 1.0


def test_dataset_label_normalized() -> None:
    label = DatasetLabel(
        dataset_name="coco",
        original_label="Truck",
        normalized_label="truck",
        confidence=0.95,
    )
    assert label.normalized_label == "truck"
    assert label.confidence == 0.95


def test_mapping_rule_generates_id() -> None:
    rule = MappingRule(
        dataset_name="coco",
        source_label="car",
        canonical_value="ground_vehicle.car",
        match_type=MatchType.EXACT,
    )
    assert rule.rule_id is not None
    assert len(rule.rule_id) > 0


def test_mapping_rule_all_fields() -> None:
    dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rule = MappingRule(
        rule_id="custom-id",
        dataset_name="coco",
        source_label="car",
        canonical_value="ground_vehicle.car",
        match_type=MatchType.EXACT,
        confidence=0.9,
        created_at=dt,
        version="2.0.0",
    )
    assert rule.rule_id == "custom-id"
    assert rule.match_type == MatchType.EXACT
    assert rule.version == "2.0.0"
    assert rule.created_at == dt


def test_mapping_conflict_requires_two_rules() -> None:
    r1 = MappingRule(
        dataset_name="coco",
        source_label="truck",
        canonical_value="ground_vehicle.truck",
        match_type=MatchType.EXACT,
    )
    r2 = MappingRule(
        dataset_name="coco",
        source_label="truck",
        canonical_value="ground_vehicle.heavy_equipment",
        match_type=MatchType.SYNONYM,
        confidence=0.7,
    )
    conflict = MappingConflict(
        conflict_type=ConflictType.CONFLICTING,
        source_label="truck",
        rules=(r1, r2),
        description="Test conflict",
    )
    assert len(conflict.rules) == 2
    assert not conflict.resolved


def test_mapping_result_defaults() -> None:
    result = MappingResult(
        source_label="car",
        canonical_value="ground_vehicle.car",
        canonical_name="Car",
        confidence=1.0,
        match_type=MatchType.EXACT,
    )
    assert result.rule_id is None
    assert result.alternatives == ()


def test_mapping_result_with_alternatives() -> None:
    result = MappingResult(
        source_label="truck",
        canonical_value="ground_vehicle.truck",
        canonical_name="Truck",
        confidence=1.0,
        match_type=MatchType.EXACT,
        rule_id="rule-1",
        alternatives=(("ground_vehicle.heavy_equipment", 0.7),),
    )
    assert result.rule_id == "rule-1"
    assert len(result.alternatives) == 1


def test_dataset_mapping_frozen() -> None:
    mapping = DatasetMapping(
        dataset_name="coco",
        dataset_version="2017",
        ontology_version="1.0.0",
    )
    assert mapping.rules == ()


def test_dataset_profile_validates() -> None:
    profile = DatasetProfile(
        dataset_name="coco",
        version="2017",
        label_count=80,
        labels=("car", "truck"),
    )
    assert profile.label_count == 80


def test_dataset_profile_optional_description() -> None:
    profile = DatasetProfile(
        dataset_name="coco",
        version="2017",
        label_count=80,
        labels=("car",),
        description="COCO 2017 dataset",
    )
    assert profile.description == "COCO 2017 dataset"


def test_mapping_statistics() -> None:
    stats = MappingStatistics(
        dataset_name="coco",
        total_labels=80,
        mapped_labels=60,
        ignored_labels=10,
        unknown_labels=10,
        coverage_percent=75.0,
        conflict_count=2,
        duplicate_alias_count=1,
        ontology_version="1.0.0",
    )
    assert stats.coverage_percent == 75.0
    assert stats.mapped_labels + stats.unknown_labels == 70


def test_mapping_version_defaults() -> None:
    v = MappingVersion(version="1.0.0", ontology_version="1.0.0")
    assert v.changelog == ""
    assert v.dataset_count == 0
    assert v.rule_count == 0


def test_ontology_resolution_empty() -> None:
    res = OntologyResolution(query="nonexistent", resolved_node=None)
    assert res.resolved_node is None
    assert res.path == ()


def test_conflict_resolution_required_fields() -> None:
    res = ConflictResolution(
        conflict_id="conflict-1",
        resolution_type=ResolutionType.FIRST_WINS,
        explanation="Kept first rule",
    )
    assert res.chosen_rule is None
    assert res.merged_value is None


def test_export_format_values() -> None:
    assert ExportFormat.JSON.value == "json"
    assert ExportFormat.YAML.value == "yaml"
    assert ExportFormat.CSV.value == "csv"


def test_match_type_all_values() -> None:
    assert MatchType.EXACT.value == "exact"
    assert MatchType.ALIAS.value == "alias"
    assert MatchType.CASE_INSENSITIVE.value == "case_insensitive"
    assert MatchType.PLURAL.value == "plural"
    assert MatchType.SYNONYM.value == "synonym"
    assert MatchType.REGEX.value == "regex"
    assert MatchType.EMBEDDING.value == "embedding"


def test_ontology_node_depth_range() -> None:
    node_root = OntologyNode(
        value="root", name="Root", category="root",
        category_name="Root", depth=0, is_leaf=False,
    )
    node_cat = OntologyNode(
        value="ground_vehicle", name="Ground Vehicle",
        category="ground_vehicle", category_name="Ground Vehicle",
        parent="root", depth=1, is_leaf=False,
    )
    node_class = OntologyNode(
        value="ground_vehicle.car", name="Car",
        category="ground_vehicle", category_name="Ground Vehicle",
        parent="ground_vehicle", depth=2, is_leaf=True,
    )
    assert node_root.depth == 0
    assert node_cat.depth == 1
    assert node_class.depth == 2


def test_mapping_result_immutable() -> None:
    result = MappingResult(
        source_label="car",
        canonical_value="ground_vehicle.car",
        canonical_name="Car",
        confidence=1.0,
        match_type=MatchType.EXACT,
    )
    try:
        result.confidence = 0.5
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        pass


def test_all_conflict_types() -> None:
    assert ConflictType.DUPLICATE == "duplicate"
    assert ConflictType.CONFLICTING == "conflicting"
    assert ConflictType.CIRCULAR == "circular"
    assert ConflictType.MISSING_NODE == "missing_node"
    assert ConflictType.AMBIGUOUS_ALIAS == "ambiguous_alias"
    assert ConflictType.UNKNOWN_LABEL == "unknown_label"


def test_all_resolution_types() -> None:
    assert ResolutionType.FIRST_WINS == "first_wins"
    assert ResolutionType.LAST_WINS == "last_wins"
    assert ResolutionType.HIGHEST_CONFIDENCE == "highest_confidence"
    assert ResolutionType.MERGE == "merge"
    assert ResolutionType.MANUAL == "manual"
