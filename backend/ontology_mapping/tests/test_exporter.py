"""Tests for MappingExporter."""

from __future__ import annotations

import json

import pytest

from backend.ontology_mapping.exporter import MappingExporter
from backend.ontology_mapping.models import (
    DatasetMapping,
    MappingRule,
    MatchType,
)


@pytest.fixture
def exporter() -> MappingExporter:
    return MappingExporter()


@pytest.fixture
def sample_mapping() -> DatasetMapping:
    return DatasetMapping(
        dataset_name="coco",
        dataset_version="2017",
        ontology_version="1.0.0",
        rules=(
            MappingRule(
                rule_id="r1",
                dataset_name="coco",
                source_label="car",
                canonical_value="ground_vehicle.car",
                match_type=MatchType.EXACT,
                confidence=1.0,
                version="1.0.0",
            ),
            MappingRule(
                rule_id="r2",
                dataset_name="coco",
                source_label="truck",
                canonical_value="ground_vehicle.truck",
                match_type=MatchType.EXACT,
                confidence=1.0,
                version="1.0.0",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_export_json(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    output = await exporter.to_json(sample_mapping)
    data = json.loads(output)
    assert data["dataset_name"] == "coco"
    assert data["dataset_version"] == "2017"
    assert len(data["rules"]) == 2


@pytest.mark.asyncio
async def test_export_json_rule_fields(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    output = await exporter.to_json(sample_mapping)
    data = json.loads(output)
    first_rule = data["rules"][0]
    assert first_rule["source_label"] == "car"
    assert first_rule["canonical_value"] == "ground_vehicle.car"
    assert first_rule["match_type"] == "exact"


@pytest.mark.asyncio
async def test_export_csv(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    output = await exporter.to_csv(sample_mapping)
    lines = output.strip().split("\n")
    assert len(lines) == 3
    header = lines[0]
    assert "source_label" in header
    assert "canonical_value" in header
    assert "match_type" in header


@pytest.mark.asyncio
async def test_export_csv_content(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    output = await exporter.to_csv(sample_mapping)
    lines = output.strip().split("\n")
    assert "car" in lines[1]
    assert "ground_vehicle.car" in lines[1]


@pytest.mark.asyncio
async def test_export_yaml(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    try:
        output = await exporter.to_yaml(sample_mapping)
    except ImportError:
        pytest.skip("PyYAML not installed")
    assert "coco" in output
    assert "ground_vehicle.car" in output


@pytest.mark.asyncio
async def test_export_json_empty_rules(exporter: MappingExporter) -> None:
    mapping = DatasetMapping(
        dataset_name="empty",
        dataset_version="1.0",
        ontology_version="1.0.0",
    )
    output = await exporter.to_json(mapping)
    data = json.loads(output)
    assert len(data["rules"]) == 0


@pytest.mark.asyncio
async def test_export_csv_empty_rules(exporter: MappingExporter) -> None:
    mapping = DatasetMapping(
        dataset_name="empty",
        dataset_version="1.0",
        ontology_version="1.0.0",
    )
    output = await exporter.to_csv(mapping)
    lines = output.strip().split("\n")
    assert len(lines) == 1


@pytest.mark.asyncio
async def test_export_json_roundtrip(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    output = await exporter.to_json(sample_mapping)
    data = json.loads(output)
    assert data["dataset_name"] == sample_mapping.dataset_name
    assert data["dataset_version"] == sample_mapping.dataset_version


@pytest.mark.asyncio
async def test_export_csv_rule_count(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    output = await exporter.to_csv(sample_mapping)
    lines = output.strip().split("\n")
    data_lines = [ln for ln in lines if ln and not ln.startswith("rule_id")]
    assert len(data_lines) == len(sample_mapping.rules)


@pytest.mark.asyncio
async def test_export_yaml_content(
    exporter: MappingExporter, sample_mapping: DatasetMapping
) -> None:
    try:
        output = await exporter.to_yaml(sample_mapping)
    except ImportError:
        pytest.skip("PyYAML not installed")
    assert "dataset_name: coco" in output
    assert "source_label: car" in output
    assert "canonical_value: ground_vehicle.car" in output
