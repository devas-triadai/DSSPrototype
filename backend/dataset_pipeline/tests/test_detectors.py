"""Tests for dataset layout detectors.

Covers all 6 well-known dataset structures plus error cases.
"""

from __future__ import annotations

from pathlib import Path

from backend.dataset_pipeline.detectors import detect_layout

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_coco(root: Path) -> None:
    (root / "train2017").mkdir(parents=True)
    (root / "val2017").mkdir()
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_train2017.json").write_text("{}")
    (root / "annotations" / "instances_val2017.json").write_text("{}")


def _make_open_images(root: Path) -> None:
    for d in ("train", "validation", "test"):
        (root / d).mkdir(parents=True)


def _make_visdrone(root: Path) -> None:
    for d in ("VisDrone2019-DET-train", "VisDrone2019-DET-val", "VisDrone2019-DET-test-dev"):
        (root / d / "images").mkdir(parents=True)
        (root / d / "annotations").mkdir(parents=True)


def _make_loveda(root: Path) -> None:
    for d in ("Train", "Val", "Test"):
        (root / d).mkdir(parents=True)


def _make_spacenet(root: Path) -> None:
    (root / "AOI_2_Vegas").mkdir(parents=True)
    (root / "AOI_3_Paris").mkdir(parents=True)


def _make_seaships(root: Path) -> None:
    for d in ("train", "val"):
        (root / d).mkdir(parents=True)


# ------------------------------------------------------------------
# Valid layouts
# ------------------------------------------------------------------


class TestDetectCOCO:
    def test_valid_coco(self, tmp_path: Path) -> None:
        _make_coco(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout is not None
        assert layout.dataset_type == "coco"
        assert sorted(d.name for d in layout.image_directories) == sorted(["train2017", "val2017"])
        assert layout.annotation_directory is not None
        assert layout.annotation_directory.name == "annotations"
        assert len(layout.annotation_files) == 2

    def test_missing_train2017(self, tmp_path: Path) -> None:
        (tmp_path / "val2017").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_train2017.json").write_text("{}")
        (tmp_path / "annotations" / "instances_val2017.json").write_text("{}")
        assert detect_layout(tmp_path) is None

    def test_missing_val2017(self, tmp_path: Path) -> None:
        (tmp_path / "train2017").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_train2017.json").write_text("{}")
        (tmp_path / "annotations" / "instances_val2017.json").write_text("{}")
        assert detect_layout(tmp_path) is None

    def test_missing_annotations_dir(self, tmp_path: Path) -> None:
        (tmp_path / "train2017").mkdir()
        (tmp_path / "val2017").mkdir()
        assert detect_layout(tmp_path) is None

    def test_missing_instances_train(self, tmp_path: Path) -> None:
        (tmp_path / "train2017").mkdir()
        (tmp_path / "val2017").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_val2017.json").write_text("{}")
        assert detect_layout(tmp_path) is None

    def test_missing_instances_val(self, tmp_path: Path) -> None:
        (tmp_path / "train2017").mkdir()
        (tmp_path / "val2017").mkdir()
        (tmp_path / "annotations").mkdir()
        (tmp_path / "annotations" / "instances_train2017.json").write_text("{}")
        assert detect_layout(tmp_path) is None


class TestDetectOpenImagesV7:
    def test_valid(self, tmp_path: Path) -> None:
        _make_open_images(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout is not None
        assert layout.dataset_type == "open_images_v7"
        dirs = sorted(d.name for d in layout.image_directories)
        assert dirs == sorted(["train", "validation", "test"])

    def test_missing_validation(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        (tmp_path / "test").mkdir()
        assert detect_layout(tmp_path) is None


class TestDetectVisDrone:
    def test_valid(self, tmp_path: Path) -> None:
        _make_visdrone(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout is not None
        assert layout.dataset_type == "visdrone"
        assert len(layout.image_directories) == 3

    def test_missing_train(self, tmp_path: Path) -> None:
        (tmp_path / "VisDrone2019-DET-val").mkdir(parents=True)
        (tmp_path / "VisDrone2019-DET-test-dev").mkdir(parents=True)
        assert detect_layout(tmp_path) is None


class TestDetectLoveDA:
    def test_valid(self, tmp_path: Path) -> None:
        _make_loveda(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout is not None
        assert layout.dataset_type == "loveda"
        assert sorted(d.name for d in layout.image_directories) == sorted(["Train", "Val", "Test"])

    def test_missing_val(self, tmp_path: Path) -> None:
        (tmp_path / "Train").mkdir()
        (tmp_path / "Test").mkdir()
        assert detect_layout(tmp_path) is None


class TestDetectSpaceNet:
    def test_valid(self, tmp_path: Path) -> None:
        _make_spacenet(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout is not None
        assert layout.dataset_type == "spacenet"
        assert len(layout.image_directories) == 2
        assert all(d.name.startswith("AOI_") for d in layout.image_directories)

    def test_no_aoi_directories(self, tmp_path: Path) -> None:
        assert detect_layout(tmp_path) is None


class TestDetectSeaShips:
    def test_valid(self, tmp_path: Path) -> None:
        _make_seaships(tmp_path)
        layout = detect_layout(tmp_path)
        assert layout is not None
        assert layout.dataset_type == "seaships"
        assert sorted(d.name for d in layout.image_directories) == sorted(["train", "val"])

    def test_missing_val(self, tmp_path: Path) -> None:
        (tmp_path / "train").mkdir()
        assert detect_layout(tmp_path) is None


# ------------------------------------------------------------------
# Unknown / invalid
# ------------------------------------------------------------------


class TestDetectUnknown:
    def test_empty_directory(self, tmp_path: Path) -> None:
        assert detect_layout(tmp_path) is None

    def random_files_no_structure(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("hello")
        assert detect_layout(tmp_path) is None

    def test_non_existent_path(self, tmp_path: Path) -> None:
        assert detect_layout(tmp_path / "does_not_exist") is None

    def test_file_path_returns_none(self, tmp_path: Path) -> None:
        f = tmp_path / "some_file.txt"
        f.write_text("data")
        assert detect_layout(f) is None
