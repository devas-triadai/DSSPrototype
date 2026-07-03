"""Tests for TrainingDatasetExporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.training.dataset_exporter import (
    TrainingDatasetExporter,
    _find_image,
)

# ------------------------------------------------------------------
# TrainingDatasetExporter — construction
# ------------------------------------------------------------------


class TestTrainingDatasetExporter:
    def test_default_output_base(self) -> None:
        exporter = TrainingDatasetExporter()
        assert str(exporter._output_base) == "datasets\\exports\\yolo"

    def test_custom_output_base(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_export"
        exporter = TrainingDatasetExporter(custom)
        assert exporter._output_base == custom

    def test_supported_types(self) -> None:
        exporter = TrainingDatasetExporter()
        types = exporter.supported_types()
        assert "coco" in types
        assert "open_images_v7" in types
        assert "visdrone" in types
        assert "loveda" in types
        assert "spacenet" in types
        assert "seaships" in types

    def test_unsupported_type(self, tmp_path: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path)
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            exporter.export("unknown_type", tmp_path)


# ------------------------------------------------------------------
# Per-dataset-type export
# ------------------------------------------------------------------


class TestCocoExport:
    def test_export_creates_output_structure(self, tmp_path: Path, coco_source: Path) -> None:
        output_base = tmp_path / "exports"
        exporter = TrainingDatasetExporter(output_base)
        result = exporter.export("coco", coco_source)

        assert result["status"] == "completed"
        assert result["dataset_name"] == "coco2017"
        assert result["class_names"] == ["person", "car", "dog"]
        assert result["train_count"] == 3
        assert result["val_count"] == 2
        assert result["annotation_count"] == 4  # 3 train + 1 val

        output_dir = Path(result["output_dir"])
        assert (output_dir / "data.yaml").exists()
        assert (output_dir / "export.json").exists()
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "dataset_info.json").exists()
        assert (output_dir / "labels" / "train" / "000000000000.txt").exists()
        assert (output_dir / "labels" / "train" / "000000000001.txt").exists()
        assert not (output_dir / "labels" / "train" / "000000000002.txt").exists()  # no annotations
        assert (output_dir / "labels" / "val" / "100000000000.txt").exists()

    def test_data_yaml_content(self, tmp_path: Path, coco_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("coco", coco_source)
        data_yaml_str = result["data_yaml"]
        assert data_yaml_str is not None
        data_yaml = Path(data_yaml_str)
        content = data_yaml.read_text(encoding="utf-8")
        assert "nc: 3" in content
        assert "person" in content
        assert "car" in content
        assert "dog" in content
        assert "images/train" in content
        assert "images/val" in content

    def test_export_json_content(self, tmp_path: Path, coco_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("coco", coco_source)
        export_dir = Path(str(result["output_dir"]))
        export_data = json.loads((export_dir / "export.json").read_text(encoding="utf-8"))
        assert export_data["class_count"] == 3
        assert export_data["train_count"] == 3
        assert export_data["val_count"] == 2
        assert export_data["total_annotations"] == 4

    def test_label_format(self, tmp_path: Path, coco_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("coco", coco_source)
        label_file = Path(result["output_dir"]) / "labels" / "train" / "000000000000.txt"
        content = label_file.read_text(encoding="utf-8").strip()
        lines = content.splitlines()
        assert len(lines) == 2  # Two annotations
        for line in lines:
            parts = line.split()
            assert len(parts) == 5
            cls_id = int(parts[0])
            assert 0 <= cls_id < 3
            cx, cy, w, h = map(float, parts[1:])
            assert 0 <= cx <= 1
            assert 0 <= cy <= 1
            assert 0 <= w <= 1
            assert 0 <= h <= 1

    def test_custom_dataset_name(self, tmp_path: Path, coco_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("coco", coco_source, dataset_name="my_coco")
        assert result["dataset_name"] == "my_coco"
        assert result["output_dir"].endswith("my_coco")

    def test_validation_errors(self, tmp_path: Path, coco_source: Path) -> None:
        (coco_source / "train2017" / "000000000000.jpg").unlink()
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("coco", coco_source)
        assert result["status"] == "failed"
        assert len(result["errors"]) > 0

    def test_source_not_found(self, tmp_path: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        with pytest.raises(FileNotFoundError):
            exporter.export("coco", tmp_path / "nonexistent")


class TestOpenImagesExport:
    def test_export_creates_output(self, tmp_path: Path, openimages_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("open_images_v7", openimages_source)
        assert result["status"] == "completed"
        assert result["train_count"] == 3
        assert result["val_count"] == 1
        assert result["annotation_count"] >= 4

    def test_class_names(self, tmp_path: Path, openimages_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("open_images_v7", openimages_source)
        assert "Cat" in result["class_names"]
        assert "Dog" in result["class_names"]
        assert "Car" in result["class_names"]


class TestVisDroneExport:
    def test_export_creates_output(self, tmp_path: Path, visdrone_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("visdrone", visdrone_source)
        assert result["status"] == "completed"
        assert result["train_count"] == 3
        assert result["val_count"] == 1

    def test_visdrone_class_names(self, tmp_path: Path, visdrone_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("visdrone", visdrone_source)
        assert "pedestrian" in result["class_names"]
        assert "car" in result["class_names"]

    def test_labels_written(self, tmp_path: Path, visdrone_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("visdrone", visdrone_source)
        output_dir = Path(result["output_dir"])
        assert (output_dir / "labels" / "train" / "000000.txt").exists()


class TestLoveDaExport:
    def test_export_creates_output(self, tmp_path: Path, loveda_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("loveda", loveda_source)
        assert result["status"] == "completed"
        assert result["train_count"] == 2  # 2 GeoJSON files with annotations
        assert result["val_count"] == 1  # 1 GeoJSON file

    def test_geojson_parsed(self, tmp_path: Path, loveda_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("loveda", loveda_source)
        assert result["annotation_count"] >= 2


class TestSpaceNetExport:
    def test_export_creates_output(self, tmp_path: Path, spacenet_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("spacenet", spacenet_source)
        assert result["status"] == "completed"
        assert result["train_count"] == 2  # 2 GeoJSON files
        assert result["val_count"] == 1  # 1 GeoJSON file

    def test_class_names(self, tmp_path: Path, spacenet_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("spacenet", spacenet_source)
        assert result["class_names"] == ["building"]


class TestSeaShipsExport:
    def test_export_creates_output(self, tmp_path: Path, seaships_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("seaships", seaships_source)
        assert result["status"] == "completed"
        assert result["train_count"] == 2  # 2 XML files with annotations
        assert result["val_count"] == 1  # 1 XML file

    def test_class_names(self, tmp_path: Path, seaships_source: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        result = exporter.export("seaships", seaships_source)
        assert result["class_names"] == ["ship"]


# ------------------------------------------------------------------
# _find_image helper
# ------------------------------------------------------------------


class TestFindImage:
    def test_finds_jpg(self, tmp_path: Path) -> None:
        (tmp_path / "test.jpg").write_text("dummy")
        assert _find_image(tmp_path, "test") == tmp_path / "test.jpg"

    def test_finds_png(self, tmp_path: Path) -> None:
        (tmp_path / "img.png").write_text("dummy")
        assert _find_image(tmp_path, "img") == tmp_path / "img.png"

    def test_returns_none_for_missing(self, tmp_path: Path) -> None:
        assert _find_image(tmp_path, "nonexistent") is None


# ------------------------------------------------------------------
# Error cases
# ------------------------------------------------------------------


class TestExportErrors:
    def test_missing_annotations_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "empty_coco"
        source.mkdir()
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        with pytest.raises(FileNotFoundError, match="training annotations not found"):
            exporter.export("coco", source)

    def test_unsupported_type(self, tmp_path: Path) -> None:
        exporter = TrainingDatasetExporter(tmp_path / "exports")
        with pytest.raises(ValueError, match="Unsupported dataset type"):
            exporter.export("xyz", tmp_path)
