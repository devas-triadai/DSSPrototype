"""Tests for the StatisticsEngine."""

import json
import tempfile
from pathlib import Path

from backend.dataset_manager.statistics import StatisticsEngine


def _create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake_image_data")


def _create_annotation(path: Path, categories: list[int] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cats = categories or [1]
    data = {
        "annotations": [
            {"id": i, "category_id": c, "bbox": [10, 20, 50, 80]}
            for i, c in enumerate(cats)
        ],
    }
    path.write_text(json.dumps(data))


def test_statistics_empty_directory() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stats = StatisticsEngine().compute(Path(tmp))
        assert stats.total_images == 0
        assert stats.total_annotations == 0
        assert stats.classes == []


def test_statistics_with_images_only() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        _create_image(d / "img002.jpg")
        stats = StatisticsEngine().compute(d)
        assert stats.total_images == 2
        assert stats.total_annotations == 0


def test_statistics_with_annotations() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        _create_annotation(d / "img001.json", categories=[1, 2, 1, 3])
        stats = StatisticsEngine().compute(d)
        assert stats.total_images == 1
        assert stats.total_annotations == 4
        assert "class_1" in stats.objects_per_class


def test_class_distribution() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_annotation(d / "ann.json", categories=[1, 1, 2, 2, 2])
        stats = StatisticsEngine().compute(d)
        assert stats.class_distribution["class_1"] == 0.4
        assert stats.class_distribution["class_2"] == 0.6


def test_class_imbalance_ratio() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_annotation(d / "ann.json", categories=[1, 1, 1, 1, 2])
        stats = StatisticsEngine().compute(d)
        assert stats.class_imbalance_ratio == 4.0


def test_dataset_completeness() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        _create_image(d / "img001.jpg")
        _create_image(d / "img002.jpg")
        _create_annotation(d / "img001.json", categories=[1])
        stats = StatisticsEngine().compute(d)
        assert stats.dataset_completeness == 0.5


def test_generated_at_is_set() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stats = StatisticsEngine().compute(Path(tmp))
        assert stats.generated_at != ""
