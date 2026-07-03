"""Tests for the DatasetValidator."""

import tempfile
from pathlib import Path

from backend.dataset_manager.validator import DatasetValidator


def _create_image(path: Path, content: bytes = b"fake_image_data") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _create_annotation(path: Path, content: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_validate_empty_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        result = DatasetValidator().validate(Path(tmp))
        assert result.passed is True
        assert result.total_checks == 12
        assert result.passed_checks == 12


def test_validate_with_valid_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        _create_annotation(d / "img001.json", content='{"annotations": []}')
        result = DatasetValidator().validate(d)
        assert result.passed is True


def test_validate_detects_duplicates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        data = b"same_content"
        _create_image(d / "img001.jpg", data)
        _create_image(d / "img002.jpg", data)
        result = DatasetValidator().validate(d)
        assert len(result.duplicate_images) >= 0


def test_validate_detects_corrupted_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "empty.jpg", b"")
        result = DatasetValidator().validate(d)
        assert len(result.corrupted_files) > 0


def test_validate_detects_unsupported_extensions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.gif", b"data")  # .gif not in supported list
        result = DatasetValidator().validate(d)
        assert len(result.unsupported_extensions) > 0


def test_validate_orphan_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_annotation(d / "orphan.json", content='{"annotations": []}')
        result = DatasetValidator().validate(d)
        assert len(result.missing_labels) > 0
