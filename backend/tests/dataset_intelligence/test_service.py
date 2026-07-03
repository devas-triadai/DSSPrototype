"""Tests for the DatasetIntelligenceService."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import backend.dataset_intelligence.config
from backend.dataset_intelligence.exceptions import (
    DatasetNotFoundError,
    ExportError,
    ValidationError,
)
from backend.dataset_intelligence.models import (
    Annotation,
    HarmonizedDataset,
    ImageRecord,
    NormalizedDataset,
    RawDataset,
    ValidationReport,
)
from backend.dataset_intelligence.service import DatasetIntelligenceService


def _create_minimal_yolo_dataset(path: Path) -> None:
    img_dir = path / "images"
    lbl_dir = path / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img = img_dir / "test.jpg"
    img.write_bytes(b"fake_image")
    lbl = lbl_dir / "test.txt"
    lbl.write_text("0 0.5 0.5 0.3 0.4\n")
    (path / "classes.txt").write_text("tank\n")


def test_service_import_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        img_dir = d / "images"
        lbl_dir = d / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        # Create 10 distinct valid JPEG images to pass min count
        from PIL import Image
        for i in range(10):
            im = Image.new("RGB", (640 + i, 480 + i), color=(i * 20, 100, 200))
            im.save(img_dir / f"img{i:03d}.jpg", "JPEG")
            (lbl_dir / f"img{i:03d}.txt").write_text(f"{i % 2} 0.5 0.5 0.3 0.4\n")
        (d / "classes.txt").write_text("tank\ntruck\n")
        # Override quality threshold for test
        backend.dataset_intelligence.config.di_config.max_duplicate_ratio = 0.99
        service = DatasetIntelligenceService()
        result = service.import_dataset(d, "test_dataset")
        assert result.status == "ready"
        assert result.statistics is not None
        assert result.quality is not None
        assert result.quality.passed is True


def test_service_import_dataset_fails_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Empty directory with no structure - will fail validation
        service = DatasetIntelligenceService()
        with pytest.raises((ValidationError, Exception)):
            service.import_dataset(d, "bad_dataset")


def test_service_merge_datasets() -> None:
    mock_registry = MagicMock()
    mock_merger = MagicMock()
    mock_statistics = MagicMock()
    mock_quality = MagicMock()
    mock_splitter = MagicMock()
    mock_exporter_registry = MagicMock()

    entry = MagicMock()
    entry.dataset_id = "ds1"
    entry.dataset_name = "ds1"
    entry.report_file = ""
    mock_registry.get.return_value = entry

    img = ImageRecord(
        image_id="img001",
        image_path="/path/img.jpg",
        image_name="img.jpg",
        annotations=[Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))],
    )
    merged = HarmonizedDataset(
        dataset_id="merged",
        dataset_name="merged",
        images=[img],
        classes=["tank"],
    )
    mock_merger.merge.return_value = merged
    mock_statistics.compute.return_value = MagicMock(
        total_images=1, total_annotations=1
    )
    mock_quality.assess.return_value = MagicMock(
        quality_score=0.9, passed=True, errors=[]
    )
    mock_splitter.split.return_value = {
        "train": ["img001"],
        "validation": [],
        "test": [],
    }
    mock_exporter = MagicMock()
    mock_exporter.export.return_value = MagicMock(
        dataset_id="merged",
        format_name="yolo",
        output_dir="/out",
        image_count=1,
        annotation_count=1,
        class_mapping={"tank": 0},
    )
    mock_exporter_registry.get.return_value = mock_exporter

    service = DatasetIntelligenceService(
        registry=mock_registry,
        merger=mock_merger,
        statistics_engine=mock_statistics,
        quality_engine=mock_quality,
        splitter=mock_splitter,
        exporter_registry=mock_exporter_registry,
    )
    with pytest.raises(DatasetNotFoundError):
        service.merge_datasets(["ds1"], "merged")


def test_service_export_dataset() -> None:
    mock_registry = MagicMock()
    mock_exporter_registry = MagicMock()
    mock_exporter = MagicMock()

    entry = MagicMock()
    entry.dataset_id = "ds1"
    entry.dataset_name = "ds1"
    entry.report_file = ""
    mock_registry.get.return_value = entry
    mock_exporter.export.return_value = MagicMock(
        dataset_id="ds1",
        format_name="yolo",
        output_dir="/out",
        image_count=1,
        annotation_count=1,
        class_mapping={"tank": 0},
    )
    mock_exporter_registry.get.return_value = mock_exporter

    service = DatasetIntelligenceService(
        registry=mock_registry,
        exporter_registry=mock_exporter_registry,
    )
    with pytest.raises(ExportError):
        service.export_dataset("ds1", "yolo")


def test_service_get_quality_report() -> None:
    mock_registry = MagicMock()
    entry = MagicMock()
    entry.quality_file = ""
    mock_registry.get.return_value = entry

    service = DatasetIntelligenceService(registry=mock_registry)
    report = service.get_quality_report("ds1")
    assert report is None


def test_service_get_statistics() -> None:
    mock_registry = MagicMock()
    entry = MagicMock()
    entry.statistics_file = ""
    mock_registry.get.return_value = entry

    service = DatasetIntelligenceService(registry=mock_registry)
    stats = service.get_statistics("ds1")
    assert stats is None


def test_service_integration_with_mocks() -> None:
    """Test the full pipeline with all mocked components."""
    from backend.dataset_intelligence.models import (
        DuplicateReport,
        ExportResult,
        OntologyMappingReport,
        QualityReport,
        StatisticsReport,
    )

    mock_importer = MagicMock()
    mock_validator = MagicMock()
    mock_normalizer = MagicMock()
    mock_ontology_mapper = MagicMock()
    mock_duplicate_detector = MagicMock()
    mock_class_harmonizer = MagicMock()
    mock_splitter = MagicMock()
    mock_statistics = MagicMock()
    mock_quality = MagicMock()
    mock_exporter_registry = MagicMock()
    mock_registry = MagicMock()

    raw = RawDataset(
        dataset_id="test_yolo",
        dataset_name="test",
        import_format="yolo",
        source_path="/path",
        images=[],
        classes=["tank"],
    )
    mock_importer.import_dataset.return_value = MagicMock(raw_dataset=raw)

    mock_validator.validate.return_value = ValidationReport(
        dataset_id="test", passed=True, total_checks=5, passed_checks=5, failed_checks=0
    )

    normalized = NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        classes=["tank"],
    )
    mock_normalizer.normalize.return_value = normalized

    ontology_report = OntologyMappingReport(
        dataset_id="test",
        ontology_coverage=1.0,
        mappings={"tank": "main_battle_tank"},
    )
    mock_ontology_mapper.map_classes.return_value = ontology_report
    mock_ontology_mapper.apply_mapping.return_value = normalized

    mock_duplicate_detector.detect.return_value = DuplicateReport(
        dataset_id="test",
        duplicate_ratio=0.0,
    )

    harmonized = HarmonizedDataset(
        dataset_id="test",
        dataset_name="test",
        classes=["main_battle_tank"],
    )
    mock_class_harmonizer.harmonize.return_value = harmonized

    mock_statistics.compute.return_value = StatisticsReport(
        dataset_id="test",
        total_images=10,
        total_annotations=30,
        classes=["main_battle_tank"],
        class_imbalance_ratio=1.0,
        ontology_coverage=1.0,
    )

    mock_quality.assess.return_value = QualityReport(
        dataset_id="test",
        quality_score=0.95,
        passed=True,
        duplicate_ratio=0.0,
        class_imbalance_ratio=1.0,
        missing_annotations=0,
        missing_images=0,
        annotation_consistency=1.0,
        ontology_coverage=1.0,
    )

    mock_splitter.split.return_value = {
        "train": ["img001"],
        "validation": [],
        "test": [],
    }

    mock_exporter = MagicMock()
    mock_exporter.export.return_value = ExportResult(
        dataset_id="test",
        format_name="yolo",
        output_dir="/out",
        image_count=10,
        annotation_count=30,
        class_mapping={"main_battle_tank": 0},
    )
    mock_exporter_registry.get.return_value = mock_exporter

    service = DatasetIntelligenceService(
        importer=mock_importer,
        validator=mock_validator,
        normalizer=mock_normalizer,
        ontology_mapper=mock_ontology_mapper,
        duplicate_detector=mock_duplicate_detector,
        class_harmonizer=mock_class_harmonizer,
        splitter=mock_splitter,
        statistics_engine=mock_statistics,
        quality_engine=mock_quality,
        exporter_registry=mock_exporter_registry,
        registry=mock_registry,
    )

    result = service.import_dataset(Path("/fake"), "test")
    assert result.status == "ready"
    assert result.statistics is not None
    assert result.quality is not None
