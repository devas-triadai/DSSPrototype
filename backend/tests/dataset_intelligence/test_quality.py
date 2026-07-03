"""Tests for the QualityEngine."""

from backend.dataset_intelligence.models import (
    Annotation,
    DuplicateReport,
    ImageRecord,
    NormalizedDataset,
    StatisticsReport,
    ValidationReport,
)
from backend.dataset_intelligence.quality import QualityEngine


def _make_dataset() -> NormalizedDataset:
    ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    img = ImageRecord(
        image_id="img001",
        image_path="/path/img.jpg",
        image_name="img.jpg",
        width=640,
        height=480,
        annotations=[ann],
    )
    return NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        images=[img],
        classes=["tank"],
    )


def _make_stats(
    dataset: NormalizedDataset,
    imbalance: float = 1.0,
    coverage: float = 1.0,
) -> StatisticsReport:
    total_anns = sum(len(img.annotations) for img in dataset.images)
    return StatisticsReport(
        dataset_id=dataset.dataset_id,
        total_images=len(dataset.images),
        total_annotations=total_anns,
        classes=dataset.classes,
        class_count=len(dataset.classes),
        class_imbalance_ratio=imbalance,
        ontology_coverage=coverage,
    )


def _make_validation() -> ValidationReport:
    return ValidationReport(
        dataset_id="test",
        passed=True,
        total_checks=10,
        passed_checks=10,
        failed_checks=0,
    )


def _make_duplicates(ratio: float = 0.0) -> DuplicateReport:
    return DuplicateReport(
        dataset_id="test",
        duplicate_ratio=ratio,
    )


def test_quality_passes() -> None:
    ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    images = [
        ImageRecord(
            image_id=f"img{i:03d}", image_path=f"/p/{i}.jpg",
            image_name=f"{i}.jpg", width=640, height=480,
            annotations=[ann],
        )
        for i in range(10)
    ]
    dataset = NormalizedDataset(
        dataset_id="test", dataset_name="test",
        images=images, classes=["tank"],
    )
    stats = StatisticsReport(
        dataset_id="test", total_images=10, total_annotations=10,
        classes=["tank"], class_imbalance_ratio=1.0, ontology_coverage=1.0,
    )
    quality = QualityEngine()
    report = quality.assess(
        dataset,
        _make_validation(),
        _make_duplicates(),
        stats,
    )
    assert report.quality_score > 0.5
    assert report.passed is True


def test_quality_fails_high_duplicates() -> None:
    dataset = _make_dataset()
    quality = QualityEngine()
    report = quality.assess(
        dataset,
        _make_validation(),
        _make_duplicates(ratio=0.5),
        _make_stats(dataset),
    )
    assert len(report.errors) > 0
    assert any("duplicate" in e.lower() for e in report.errors)


def test_quality_fails_high_imbalance() -> None:
    dataset = _make_dataset()
    quality = QualityEngine()
    report = quality.assess(
        dataset,
        _make_validation(),
        _make_duplicates(),
        _make_stats(dataset, imbalance=20.0),
    )
    assert len(report.warnings) > 0
    assert any("imbalance" in w.lower() for w in report.warnings)


def test_quality_reports_missing_images() -> None:
    dataset = _make_dataset()
    validation = ValidationReport(
        dataset_id="test",
        passed=True,
        missing_images=["img001"],
        total_checks=10,
        passed_checks=10,
        failed_checks=0,
    )
    quality = QualityEngine()
    report = quality.assess(
        dataset,
        validation,
        _make_duplicates(),
        _make_stats(dataset),
    )
    assert report.missing_images > 0


def test_quality_low_coverage() -> None:
    dataset = _make_dataset()
    quality = QualityEngine()
    report = quality.assess(
        dataset,
        _make_validation(),
        _make_duplicates(),
        _make_stats(dataset, coverage=0.1),
    )
    assert any("coverage" in w.lower() for w in report.warnings)
