"""Tests for dataset exporters."""

import tempfile
from pathlib import Path

from backend.dataset_intelligence.exporter import (
    CocoExporter,
    ExporterRegistry,
    VocExporter,
    YoloExporter,
)
from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
)


def _make_dataset() -> NormalizedDataset:
    ann = Annotation(
        class_name="tank",
        normalized_class="main_battle_tank",
        ontology_class="main_battle_tank",
        bbox=(0.1, 0.2, 0.5, 0.6),
    )
    img = ImageRecord(
        image_id="img001",
        image_path="/fake/path/img001.jpg",
        image_name="img001.jpg",
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


def test_yolo_exporter() -> None:
    dataset = _make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        exporter = YoloExporter()
        result = exporter.export(dataset, Path(tmp))
        assert result.format_name == "yolo"
        assert result.image_count == 1
        assert result.annotation_count == 1
        assert "main_battle_tank" in result.class_mapping


def test_yolo_exporter_with_splits() -> None:
    dataset = _make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        exporter = YoloExporter()
        splits = {"train": ["img001"], "val": []}
        result = exporter.export(
            dataset,
            Path(tmp),
            class_mapping={"main_battle_tank": 0},
            splits=splits,
        )
        assert result.image_count == 1
        assert (Path(tmp) / "train" / "images").exists()
        assert (Path(tmp) / "train" / "labels").exists()


def test_coco_exporter() -> None:
    dataset = _make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        exporter = CocoExporter()
        result = exporter.export(dataset, Path(tmp))
        assert result.format_name == "coco"
        assert result.image_count == 1
        instances_file = Path(tmp) / "instances.json"
        assert instances_file.exists()


def test_voc_exporter() -> None:
    dataset = _make_dataset()
    with tempfile.TemporaryDirectory() as tmp:
        exporter = VocExporter()
        result = exporter.export(dataset, Path(tmp))
        assert result.format_name == "voc"
        assert result.image_count == 1
        annotations_dir = Path(tmp) / "Annotations"
        assert annotations_dir.exists()
        xml_files = list(annotations_dir.glob("*.xml"))
        assert len(xml_files) == 1


def test_exporter_registry_defaults() -> None:
    registry = ExporterRegistry()
    formats = registry.list_formats()
    assert "yolo" in formats
    assert "coco" in formats
    assert "voc" in formats


def test_exporter_registry_get() -> None:
    registry = ExporterRegistry()
    exporter = registry.get("yolo")
    assert exporter.format_name == "yolo"


def test_exporter_registry_get_unknown() -> None:
    registry = ExporterRegistry()
    from backend.dataset_intelligence.exceptions import ExportError

    try:
        registry.get("unknown")
        assert False
    except ExportError:
        assert True


def test_exporter_registry_register_custom() -> None:
    registry = ExporterRegistry()
    custom = YoloExporter()
    registry.register(custom)
    assert registry.get("yolo") is custom
