"""Tests for the DatasetLoader."""

import tempfile
from pathlib import Path

from backend.dataset_manager.loader import DatasetLoader


def _create_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("")


def test_load_images_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        images = DatasetLoader().load_images(Path(tmp))
        assert images == []


def test_load_images_finds_jpg() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_file(d / "img001.jpg")
        _create_file(d / "img002.jpg")
        images = DatasetLoader().load_images(d)
        assert len(images) == 2


def test_load_images_filters_by_extension() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_file(d / "img001.jpg")
        _create_file(d / "img002.png")
        _create_file(d / "readme.txt")
        images = DatasetLoader().load_images(d)
        assert len(images) == 2


def test_load_images_recursive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_file(d / "sub1" / "img001.jpg")
        _create_file(d / "sub2" / "img002.png")
        images = DatasetLoader().load_images(d)
        assert len(images) == 2


def test_load_annotations_finds_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_file(d / "ann001.json")
        _create_file(d / "ann002.xml")
        annotations = DatasetLoader().load_annotations(d)
        assert len(annotations) >= 1
