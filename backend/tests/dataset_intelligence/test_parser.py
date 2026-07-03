"""Tests for format parsers."""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from backend.dataset_intelligence.exceptions import FormatDetectionError
from backend.dataset_intelligence.parser import (
    CocoParser,
    CsvParser,
    FormatParserRegistry,
    GeoJsonParser,
    JsonParser,
    VocParser,
    YoloParser,
)


def _create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake_image_data")


def test_yolo_parser_with_class_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        img_dir = d / "images"
        lbl_dir = d / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        _create_image(img_dir / "img001.jpg")
        (lbl_dir / "img001.txt").write_text("0 0.5 0.5 0.3 0.4\n")
        (d / "classes.txt").write_text("tank\n")
        parser = YoloParser()
        result = parser.parse(d)
        assert result.import_format == "yolo"
        assert len(result.images) == 1
        assert result.classes == ["tank"]


def test_yolo_parser_no_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        parser = YoloParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == []


def test_coco_parser() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        coco_data = {
            "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [100, 150, 200, 300],
                }
            ],
            "categories": [{"id": 1, "name": "tank"}],
        }
        (d / "instances.json").write_text(json.dumps(coco_data))
        parser = CocoParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == ["tank"]


def test_coco_parser_no_annotation_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        parser = CocoParser()
        try:
            parser.parse(d)
            assert False, "Should raise FormatDetectionError"
        except FormatDetectionError:
            assert True


def test_voc_parser() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = "img001.jpg"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = "640"
        ET.SubElement(size, "height").text = "480"
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = "tank"
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = "100"
        ET.SubElement(bbox, "ymin").text = "150"
        ET.SubElement(bbox, "xmax").text = "300"
        ET.SubElement(bbox, "ymax").text = "450"
        tree = ET.ElementTree(root)
        tree.write(str(d / "img001.xml"), encoding="unicode", xml_declaration=True)
        parser = VocParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == ["tank"]


def test_csv_parser() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        csv_content = "image_filename,class_name,x_min,y_min,x_max,y_max,width,height\n"
        csv_content += "img001.jpg,tank,100,150,300,450,640,480\n"
        (d / "annotations.csv").write_text(csv_content)
        parser = CsvParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == ["tank"]


def test_csv_parser_no_csv() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        parser = CsvParser()
        try:
            parser.parse(d)
            assert False
        except FormatDetectionError:
            assert True


def test_json_parser() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        json_data = {
            "images": [
                {
                    "filename": "img001.jpg",
                    "width": 640,
                    "height": 480,
                    "annotations": [{"class_name": "tank", "bbox": [0.1, 0.2, 0.5, 0.6]}],
                }
            ]
        }
        (d / "data.json").write_text(json.dumps(json_data))
        parser = JsonParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == ["tank"]


def test_json_parser_array() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        json_data = [
            {
                "filename": "img001.jpg",
                "width": 640,
                "height": 480,
                "annotations": [{"class_name": "truck", "bbox": [0.1, 0.2, 0.5, 0.6]}],
            }
        ]
        (d / "data.json").write_text(json.dumps(json_data))
        parser = JsonParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == ["truck"]


def test_geojson_parser() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        geojson_data = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "image_filename": "img001.jpg",
                        "class_name": "tank",
                        "bbox": [0.1, 0.2, 0.5, 0.6],
                        "width": 640,
                        "height": 480,
                        "bbox_format": "xyxy_normalized",
                    },
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                }
            ],
        }
        (d / "data.geojson").write_text(json.dumps(geojson_data))
        parser = GeoJsonParser()
        result = parser.parse(d)
        assert len(result.images) == 1
        assert result.classes == ["tank"]


def test_format_detection_yolo() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "images").mkdir()
        (d / "labels").mkdir()
        registry = FormatParserRegistry()
        fmt = registry.detect_format(d)
        assert fmt == "yolo"


def test_format_detection_voc() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "test.xml").write_text("<annotation></annotation>")
        registry = FormatParserRegistry()
        fmt = registry.detect_format(d)
        assert fmt == "voc"


def test_format_detection_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        registry = FormatParserRegistry()
        try:
            registry.detect_format(d)
            assert False
        except FormatDetectionError:
            assert True


def test_get_parser_registered() -> None:
    registry = FormatParserRegistry()
    parser = registry.get_parser("yolo")
    assert parser is not None
    assert "yolo" in parser.supported_formats


def test_get_parser_unregistered() -> None:
    registry = FormatParserRegistry()
    try:
        registry.get_parser("unknown_format")
        assert False
    except FormatDetectionError:
        assert True


def test_register_custom_parser() -> None:
    registry = FormatParserRegistry()
    custom = YoloParser()
    registry.register(custom)
    assert registry.get_parser("yolo") is custom
