"""Tests for DatasetFormatResolver.

Covers layout-based, file-based, and heuristic format resolution.
"""

from __future__ import annotations

from pathlib import Path

from backend.dataset_pipeline.models import DatasetLayout
from backend.dataset_pipeline.resolver import DatasetFormatResolver


class TestResolveFromLayout:
    def test_coco_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="coco",
            root_path=Path("/data/coco"),
            image_directories=[Path("/data/coco/train2017"), Path("/data/coco/val2017")],
            annotation_directory=Path("/data/coco/annotations"),
            annotation_files=[
                Path("/data/coco/annotations/instances_train2017.json"),
                Path("/data/coco/annotations/instances_val2017.json"),
            ],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/coco")) == "coco_json"

    def test_open_images_v7_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="open_images_v7",
            root_path=Path("/data/oi"),
            image_directories=[Path("/data/oi/train"), Path("/data/oi/validation")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/oi")) == "coco_json"

    def test_visdrone_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="visdrone",
            root_path=Path("/data/visdrone"),
            image_directories=[Path("/data/visdrone/VisDrone2019-DET-train/images")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/visdrone")) == "coco_json"

    def test_loveda_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="loveda",
            root_path=Path("/data/loveda"),
            image_directories=[Path("/data/loveda/Train")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/loveda")) == "coco_json"

    def test_spacenet_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="spacenet",
            root_path=Path("/data/spacenet"),
            image_directories=[Path("/data/spacenet/AOI_2_Vegas")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/spacenet")) == "coco_json"

    def test_seaships_layout(self) -> None:
        layout = DatasetLayout(
            dataset_type="seaships",
            root_path=Path("/data/seaships"),
            image_directories=[Path("/data/seaships/train"), Path("/data/seaships/val")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/seaships")) == "coco_json"


class TestResolveFromAnnotationFiles:
    def test_json_annotation_file(self) -> None:
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=Path("/data/custom"),
            image_directories=[Path("/data/custom")],
            annotation_files=[Path("/data/custom/_annotations.json")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/custom")) == "coco_json"

    def test_xml_annotation_files(self) -> None:
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=Path("/data/voc"),
            image_directories=[Path("/data/voc/JPEGImages")],
            annotation_files=[Path("/data/voc/Annotations/000001.xml")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/voc")) == "pascal_voc"

    def test_txt_annotation_files(self) -> None:
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=Path("/data/yolo"),
            image_directories=[Path("/data/yolo/images")],
            annotation_files=[Path("/data/yolo/labels/000001.txt")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/yolo")) == "yolo_txt"

    def test_yaml_annotation_file(self) -> None:
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=Path("/data/yolo"),
            image_directories=[Path("/data/yolo")],
            annotation_files=[Path("/data/yolo/data.yaml")],
        )
        assert DatasetFormatResolver.resolve(layout, Path("/data/yolo")) == "yolo_txt"

    def test_annotation_directory_with_json(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "annotations"
        ann_dir.mkdir()
        (ann_dir / "instances.json").write_text("{}")
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=tmp_path,
            image_directories=[tmp_path],
            annotation_directory=ann_dir,
            annotation_files=[],
        )
        assert DatasetFormatResolver.resolve(layout, tmp_path) == "coco_json"

    def test_annotation_directory_with_xml(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "Annotations"
        ann_dir.mkdir()
        (ann_dir / "000001.xml").write_text("<annotation></annotation>")
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=tmp_path,
            image_directories=[tmp_path],
            annotation_directory=ann_dir,
            annotation_files=[],
        )
        assert DatasetFormatResolver.resolve(layout, tmp_path) == "pascal_voc"

    def test_annotation_directory_with_txt(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "labels"
        ann_dir.mkdir()
        (ann_dir / "000001.txt").write_text("0 0.5 0.5 0.2 0.3")
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=tmp_path,
            image_directories=[tmp_path],
            annotation_directory=ann_dir,
            annotation_files=[],
        )
        assert DatasetFormatResolver.resolve(layout, tmp_path) == "yolo_txt"


class TestResolveFromNameHeuristics:
    def test_coco_in_path_name(self) -> None:
        assert (
            DatasetFormatResolver.resolve(
                None,
                Path("/data/coco2017"),
            )
            == "coco_json"
        )

    def test_yolo_in_path_name(self) -> None:
        assert (
            DatasetFormatResolver.resolve(
                None,
                Path("/data/yolo_dataset"),
            )
            == "yolo_txt"
        )

    def test_darknet_in_path_name(self) -> None:
        assert (
            DatasetFormatResolver.resolve(
                None,
                Path("/data/darknet_data"),
            )
            == "yolo_txt"
        )

    def test_voc_in_path_name(self) -> None:
        assert (
            DatasetFormatResolver.resolve(
                None,
                Path("/data/VOCdevkit"),
            )
            == "pascal_voc"
        )

    def test_pascal_in_path_name(self) -> None:
        assert (
            DatasetFormatResolver.resolve(
                None,
                Path("/data/pascal_voc"),
            )
            == "pascal_voc"
        )

    def test_default_to_coco_json(self) -> None:
        assert (
            DatasetFormatResolver.resolve(
                None,
                Path("/data/random_dataset"),
            )
            == "coco_json"
        )

    def test_json_file_path(self, tmp_path: Path) -> None:
        f = tmp_path / "annotations.json"
        f.write_text("{}")
        assert DatasetFormatResolver.resolve(None, f) == "coco_json"

    def test_yaml_file_path(self, tmp_path: Path) -> None:
        f = tmp_path / "data.yaml"
        f.write_text("train: ./images")
        assert DatasetFormatResolver.resolve(None, f) == "yolo_txt"


class TestEdgeCases:
    def test_none_layout_no_images(self, tmp_path: Path) -> None:
        src = tmp_path / "empty_dir"
        src.mkdir()
        assert DatasetFormatResolver.resolve(None, src) == "coco_json"

    def test_empty_annotation_files(self, tmp_path: Path) -> None:
        ann_dir = tmp_path / "ann"
        ann_dir.mkdir()
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=tmp_path,
            image_directories=[tmp_path],
            annotation_directory=ann_dir,
            annotation_files=[],
        )
        assert DatasetFormatResolver.resolve(layout, tmp_path) == "coco_json"

    def test_unknown_type_no_annotation_files(self, tmp_path: Path) -> None:
        layout = DatasetLayout(
            dataset_type="unknown",
            root_path=tmp_path,
            image_directories=[tmp_path],
            annotation_files=[],
        )
        assert DatasetFormatResolver.resolve(layout, tmp_path) == "coco_json"
