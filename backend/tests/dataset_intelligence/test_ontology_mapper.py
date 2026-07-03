"""Tests for the OntologyMapper."""

from unittest.mock import MagicMock

from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
)
from backend.dataset_intelligence.ontology_mapper import OntologyMapper


def _make_normalized_dataset() -> NormalizedDataset:
    ann = Annotation(
        class_name="tank",
        normalized_class="main_battle_tank",
        bbox=(0.1, 0.2, 0.5, 0.6),
    )
    img = ImageRecord(
        image_id="img001",
        image_path="/path/img.jpg",
        image_name="img.jpg",
        width=640,
        height=480,
        annotations=[ann],
    )
    return NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        images=[img],
        classes=["main_battle_tank"],
        class_mapping={"tank": "main_battle_tank"},
    )


def test_map_classes_with_mock_service() -> None:
    mock_service = MagicMock()
    mock_service.process.return_value.canonical_concept = "main_battle_tank"
    mock_service.get_version.return_value = "1.0.0"

    mapper = OntologyMapper(ontology_service=mock_service)
    dataset = _make_normalized_dataset()
    report = mapper.map_classes(dataset)
    assert "main_battle_tank" in report.mappings.values()
    assert report.ontology_coverage == 1.0
    assert report.ontology_version == "1.0.0"


def test_map_classes_partial_coverage() -> None:
    mock_service = MagicMock()
    mock_service.process.return_value.canonical_concept = None
    mock_service.get_version.return_value = "1.0.0"

    mapper = OntologyMapper(ontology_service=mock_service)
    dataset = _make_normalized_dataset()
    report = mapper.map_classes(dataset)
    assert report.mappings == {}
    assert report.unmapped_classes == ["main_battle_tank"]
    assert report.ontology_coverage == 0.0


def test_map_classes_service_error() -> None:
    mock_service = MagicMock()
    mock_service.process.side_effect = Exception("Service error")
    mock_service.get_version.return_value = "1.0.0"

    mapper = OntologyMapper(ontology_service=mock_service)
    dataset = _make_normalized_dataset()
    report = mapper.map_classes(dataset)
    assert report.mappings == {}
    assert "main_battle_tank" in report.unmapped_classes


def test_apply_mapping() -> None:
    mock_service = MagicMock()
    mock_service.process.return_value.canonical_concept = "main_battle_tank"
    mock_service.get_version.return_value = "1.0.0"

    mapper = OntologyMapper(ontology_service=mock_service)
    dataset = _make_normalized_dataset()
    report = mapper.map_classes(dataset)
    updated = mapper.apply_mapping(dataset, report)
    assert updated.images[0].annotations[0].ontology_class == "main_battle_tank"
