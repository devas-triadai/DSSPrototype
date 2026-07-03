"""Tests for the ClassHarmonizer."""

from backend.dataset_intelligence.class_harmonizer import ClassHarmonizer
from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
    OntologyMappingReport,
)


def _make_normalized_dataset(classes: list[str]) -> NormalizedDataset:
    anns = [Annotation(class_name=c, bbox=(0.1, 0.2, 0.5, 0.6)) for c in classes]
    img = ImageRecord(
        image_id="img001",
        image_path="/path/img.jpg",
        image_name="img.jpg",
        width=640,
        height=480,
        annotations=anns,
    )
    return NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        images=[img],
        classes=classes,
    )


def test_harmonize_all_mapped() -> None:
    dataset = _make_normalized_dataset(["tank"])
    mapping = OntologyMappingReport(
        dataset_id="test",
        mappings={"tank": "main_battle_tank"},
        ontology_coverage=1.0,
        ontology_version="1.0.0",
    )
    harmonizer = ClassHarmonizer()
    result = harmonizer.harmonize(dataset, mapping)
    assert result.classes == ["main_battle_tank"]
    assert result.harmonization_mapping == {"tank": "main_battle_tank"}


def test_harmonize_partial_mapping() -> None:
    dataset = _make_normalized_dataset(["tank", "car"])
    mapping = OntologyMappingReport(
        dataset_id="test",
        mappings={"tank": "main_battle_tank"},
        ontology_coverage=0.5,
        ontology_version="1.0.0",
    )
    harmonizer = ClassHarmonizer()
    result = harmonizer.harmonize(dataset, mapping)
    assert "main_battle_tank" in result.classes
    assert "car" in result.classes
    assert result.harmonization_mapping["car"] == "car"


def test_build_harmonization_mapping() -> None:
    dataset = _make_normalized_dataset(["tank", "MBT"])
    mapping = OntologyMappingReport(
        dataset_id="test",
        mappings={"tank": "main_battle_tank", "MBT": "main_battle_tank"},
        ontology_coverage=1.0,
        ontology_version="1.0.0",
    )
    harmonizer = ClassHarmonizer()
    hmap = harmonizer.build_harmonization_mapping(dataset.classes, mapping)
    assert hmap["tank"] == "main_battle_tank"
    assert hmap["MBT"] == "main_battle_tank"
