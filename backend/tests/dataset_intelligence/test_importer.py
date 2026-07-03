"""Tests for the DatasetImporter."""

import tempfile
from pathlib import Path

import pytest

from backend.dataset_intelligence.exceptions import ImportError
from backend.dataset_intelligence.importer import DatasetImporter


def test_import_dataset_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "images").mkdir()
        (d / "labels").mkdir()
        img = d / "images" / "test.jpg"
        img.write_bytes(b"fake")
        lab = d / "labels" / "test.txt"
        lab.write_text("0 0.5 0.5 0.3 0.4\n")
        (d / "classes.txt").write_text("tank\n")

        importer = DatasetImporter()
        result = importer.import_dataset(d, "my_dataset")
        assert result.status == "validated"
        assert result.import_format == "yolo"
        assert result.dataset_name == "my_dataset"


def test_import_dataset_with_format_hint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "images").mkdir()
        (d / "labels").mkdir()
        img = d / "images" / "test.jpg"
        img.write_bytes(b"fake")
        lab = d / "labels" / "test.txt"
        lab.write_text("0 0.5 0.5 0.3 0.4\n")
        (d / "classes.txt").write_text("tank\n")

        importer = DatasetImporter()
        result = importer.import_dataset(d, "my_dataset", format_hint="yolo")
        assert result.import_format == "yolo"


def test_import_dataset_invalid_path() -> None:
    importer = DatasetImporter()
    with pytest.raises(ImportError):
        importer.import_dataset(Path("/nonexistent/path"), "bad")


def test_import_dataset_empty_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        importer = DatasetImporter()
        with pytest.raises(ImportError):
            importer.import_dataset(d, "empty")
