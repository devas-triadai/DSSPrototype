"""Tests for the DatasetLoader."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.training.dataset_loader import DatasetLoader
from backend.training.exceptions import DatasetNotReadyError


def _make_loader(report: dict | None = None, report_name: str = "coco_1.0.json") -> tuple[DatasetLoader, Path]:  # noqa: E501
    tmp = Path(tempfile.mkdtemp())
    reports_dir = tmp / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    if report is not None:
        (reports_dir / report_name).write_text(json.dumps(report))
    return DatasetLoader(reports_dir=reports_dir), tmp


def _production_ready_report() -> dict:
    return {
        "dataset_name": "coco",
        "dataset_version": "1.0",
        "pipeline_version": "2.0.0",
        "production_ready": True,
        "overall_score": {"production_ready": True},
        "class_names": ["person", "car", "dog"],
        "num_classes": 3,
        "total_images": 1000,
        "total_annotations": 5000,
    }


def test_load_production_ready_dataset() -> None:
    report = _production_ready_report()
    loader, _ = _make_loader(report)
    result = loader.load_dataset("coco", "1.0")
    assert result.dataset_name == "coco"
    assert result.production_ready is True
    assert result.num_classes == 3
    assert result.total_images == 1000


def test_load_dataset_no_report_raises() -> None:
    loader, _ = _make_loader()
    with pytest.raises(DatasetNotReadyError, match="No quality report found"):
        loader.load_dataset("nonexistent")


def test_load_dataset_not_production_ready_raises() -> None:
    report = _production_ready_report()
    report["production_ready"] = False
    loader, _ = _make_loader(report)
    with pytest.raises(DatasetNotReadyError, match="not production ready"):
        loader.load_dataset("coco", "1.0")


def test_load_dataset_without_version() -> None:
    report = _production_ready_report()
    loader, _ = _make_loader(report, "coco.json")
    result = loader.load_dataset("coco")
    assert result.dataset_name == "coco"
    assert result.production_ready is True


def test_list_available_datasets() -> None:
    report = _production_ready_report()
    loader, _ = _make_loader(report)
    datasets = loader.list_available_datasets()
    assert len(datasets) >= 1
    assert datasets[0].dataset_name == "coco"


def test_list_available_datasets_excludes_not_ready() -> None:
    report = _production_ready_report()
    report["production_ready"] = False
    report["overall_score"]["production_ready"] = False
    loader, _ = _make_loader(report)
    datasets = loader.list_available_datasets()
    assert len(datasets) == 0


def test_list_available_datasets_empty_dir() -> None:
    loader, _ = _make_loader()
    assert loader.list_available_datasets() == []


def test_load_dataset_with_version() -> None:
    report = _production_ready_report()
    loader, _ = _make_loader(report, "coco_1.0.json")
    result = loader.load_dataset("coco", "1.0")
    assert result.dataset_version == "1.0"


def test_load_dataset_report_with_overall_score() -> None:
    report = {
        "dataset_name": "voc",
        "dataset_version": "2.0",
        "production_ready": True,
        "overall_score": {"production_ready": True},
        "class_names": ["aeroplane", "bicycle"],
        "num_classes": 2,
        "total_images": 500,
        "total_annotations": 2000,
    }
    loader, _ = _make_loader(report, "voc_2.0.json")
    result = loader.load_dataset("voc", "2.0")
    assert result.production_ready is True
    assert result.num_classes == 2


def test_load_dataset_report_with_overall_score_not_ready() -> None:
    report = {
        "dataset_name": "voc",
        "dataset_version": "2.0",
        "production_ready": False,
        "overall_score": {"production_ready": False},
    }
    loader, _ = _make_loader(report, "voc_2.0.json")
    with pytest.raises(DatasetNotReadyError):
        loader.load_dataset("voc", "2.0")


def test_load_dataset_quality_report_version() -> None:
    report = _production_ready_report()
    report["pipeline_version"] = "3.1.0"
    loader, _ = _make_loader(report)
    result = loader.load_dataset("coco", "1.0")
    assert result.quality_report_version == "3.1.0"


def test_list_available_multiple_datasets() -> None:
    tmp = Path(tempfile.mkdtemp())
    reports_dir = tmp / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    r1 = {"dataset_name": "coco", "dataset_version": "1.0", "overall_score": {"production_ready": True}}  # noqa: E501
    r2 = {"dataset_name": "voc", "dataset_version": "2.0", "overall_score": {"production_ready": True}}  # noqa: E501
    (reports_dir / "coco.json").write_text(json.dumps(r1))
    (reports_dir / "voc.json").write_text(json.dumps(r2))
    loader = DatasetLoader(reports_dir=reports_dir)
    datasets = loader.list_available_datasets()
    assert len(datasets) == 2


def test_dataset_paths_on_result() -> None:
    report = _production_ready_report()
    loader, _ = _make_loader(report)
    result = loader.load_dataset("coco", "1.0")
    assert "train" in result.train_path
    assert "val" in result.val_path
    assert "test" in result.test_path
    assert "yolo" in result.train_path
