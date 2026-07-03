"""Tests for the QualityEngine."""

from backend.dataset_manager.models import DatasetStatistics, DatasetValidation
from backend.dataset_manager.quality import QualityEngine


def test_assess_perfect_dataset() -> None:
    stats = DatasetStatistics(
        dataset_id="perfect",
        total_images=100,
        total_annotations=100,
        class_count=5,
        classes=["a", "b", "c", "d", "e"],
        objects_per_class={"a": 20, "b": 20, "c": 20, "d": 20, "e": 20},
        class_distribution={"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2},
        class_imbalance_ratio=1.0,
        average_objects_per_image=1.0,
        dataset_completeness=1.0,
        coverage_score=1.0,
    )
    validation = DatasetValidation(
        dataset_id="perfect",
        passed=True,
        total_checks=12,
        passed_checks=12,
        failed_checks=0,
    )
    quality = QualityEngine().assess(stats, validation)
    assert quality.quality_score > 0.9
    assert quality.errors == []


def test_assess_empty_dataset() -> None:
    stats = DatasetStatistics(dataset_id="empty")
    validation = DatasetValidation(
        dataset_id="empty",
        passed=False,
        total_checks=12,
        passed_checks=0,
        failed_checks=12,
    )
    quality = QualityEngine().assess(stats, validation)
    assert quality.quality_score <= 0.5
    assert len(quality.errors) > 0


def test_assess_with_warnings() -> None:
    stats = DatasetStatistics(
        dataset_id="warn",
        total_images=10,
        total_annotations=5,
        class_count=1,
        classes=["x"],
        objects_per_class={"x": 5},
        class_distribution={"x": 1.0},
        class_imbalance_ratio=0.0,
        dataset_completeness=0.5,
        coverage_score=0.0,
    )
    validation = DatasetValidation(
        dataset_id="warn",
        passed=False,
        missing_labels=["ann.json"],
        duplicate_images=[],
        duplicate_annotations=[],
        total_checks=12,
        passed_checks=8,
        failed_checks=4,
    )
    quality = QualityEngine().assess(stats, validation)
    assert quality.missing_labels > 0
    assert quality.quality_score < 1.0


def test_quality_score_range() -> None:
    stats = DatasetStatistics(dataset_id="range")
    validation = DatasetValidation(
        dataset_id="range", total_checks=12, passed_checks=6, failed_checks=6,
    )
    quality = QualityEngine().assess(stats, validation)
    assert 0.0 <= quality.quality_score <= 1.0


def test_generated_at_is_set() -> None:
    stats = DatasetStatistics(dataset_id="time")
    validation = DatasetValidation(
        dataset_id="time", total_checks=12, passed_checks=12, failed_checks=0,
    )
    quality = QualityEngine().assess(stats, validation)
    assert quality.generated_at != ""
