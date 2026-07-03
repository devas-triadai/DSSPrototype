"""Tests for dataset exporters."""

import json
import tempfile
from pathlib import Path

from backend.dataset_manager.exporter import CocoExporter, PascalVocExporter, YoloExporter


def _make_real_images(tmp: Path, *names: str) -> list[Path]:
    paths = []
    for n in names:
        p = tmp / n
        p.write_bytes(b"fake_image")
        paths.append(p)
    return paths


def test_yolo_exporter_format_name() -> None:
    assert YoloExporter().format_name == "yolo"


def test_yolo_exporter_creates_directories() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "yolo_out"
        images = _make_real_images(tmp_path, "img.jpg")
        YoloExporter().export(
            images=images,
            annotations=[],
            output_dir=out,
        )
        assert (out / "images").is_dir()
        assert (out / "labels").is_dir()


def test_yolo_exporter_creates_data_yaml() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "yaml_test"
        images = _make_real_images(tmp_path, "img.jpg")
        YoloExporter().export(
            images=images,
            annotations=[],
            output_dir=out,
            class_mapping={"person": 0, "car": 1},
        )
        yaml_file = out / "data.yaml"
        assert yaml_file.exists()
        content = yaml_file.read_text()
        assert "nc: 2" in content


def test_coco_exporter_format_name() -> None:
    assert CocoExporter().format_name == "coco"


def test_coco_exporter_creates_annotations_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "coco_out"
        images = _make_real_images(tmp_path, "img001.jpg")
        CocoExporter().export(
            images=images,
            annotations=[],
            output_dir=out,
            class_mapping={"cat": 0},
        )
        assert (out / "annotations.json").exists()
        data = json.loads((out / "annotations.json").read_text())
        assert "images" in data
        assert "categories" in data
        assert data["categories"] == [{"id": 0, "name": "cat"}]


def test_voc_exporter_format_name() -> None:
    assert PascalVocExporter().format_name == "voc"


def test_voc_exporter_creates_xml() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        out = tmp_path / "voc_out"
        images = _make_real_images(tmp_path, "img001.jpg")
        PascalVocExporter().export(
            images=images,
            annotations=[],
            output_dir=out,
        )
        xml_file = out / "img001.xml"
        assert xml_file.exists()
        content = xml_file.read_text()
        assert "<annotation>" in content
        assert "<filename>img001.jpg</filename>" in content
