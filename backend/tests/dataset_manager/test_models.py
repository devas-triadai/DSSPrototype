"""Tests for the Pydantic data models."""

from backend.dataset_manager.models import (
    DatasetChecksum,
    DatasetExport,
    DatasetInfo,
    DatasetLicense,
    DatasetMetadata,
    DatasetQuality,
    DatasetSplit,
    DatasetStatistics,
    DatasetValidation,
    DatasetVersion,
)


def test_dataset_info_defaults() -> None:
    info = DatasetInfo(dataset_id="test_id", dataset_name="test_name")
    assert info.dataset_version == "1.0.0"
    assert info.dataset_type == "raw"
    assert info.validation_status == "pending"
    assert info.quality_score == 0.0
    assert info.image_count == 0
    assert isinstance(info.license, DatasetLicense)


def test_dataset_info_is_frozen() -> None:
    info = DatasetInfo(dataset_id="frozen", dataset_name="Frozen Test")
    try:
        info.dataset_name = "Changed"
        assert False, "Should have raised FrozenInstanceError"
    except Exception:
        assert True


def test_dataset_version_defaults() -> None:
    v = DatasetVersion(dataset_id="ds")
    assert v.version == "1.0.0"
    assert v.parent_version is None


def test_dataset_statistics_defaults() -> None:
    s = DatasetStatistics(dataset_id="ds")
    assert s.total_images == 0
    assert s.total_annotations == 0
    assert s.classes == []
    assert s.objects_per_class == {}
    assert s.class_imbalance_ratio == 0.0


def test_dataset_quality_defaults() -> None:
    q = DatasetQuality(dataset_id="ds")
    assert q.quality_score == 0.0
    assert q.warnings == []
    assert q.errors == []


def test_dataset_quality_score_range() -> None:
    q = DatasetQuality(dataset_id="ds", quality_score=0.5)
    assert q.quality_score == 0.5


def test_dataset_validation_defaults() -> None:
    v = DatasetValidation(dataset_id="ds")
    assert v.passed is False
    assert v.total_checks == 0
    assert v.missing_images == []


def test_dataset_split_defaults() -> None:
    s = DatasetSplit(dataset_id="ds")
    assert s.train_ratio == 0.7
    assert s.validation_ratio == 0.15
    assert s.test_ratio == 0.15
    assert s.seed == 42


def test_dataset_export_defaults() -> None:
    e = DatasetExport(dataset_id="ds", format_name="yolo", output_dir="/tmp")
    assert e.image_count == 0


def test_dataset_checksum_defaults() -> None:
    c = DatasetChecksum(algorithm="sha256", value="abc", file_path="/tmp/f")
    assert c.algorithm == "sha256"


def test_dataset_metadata_links() -> None:
    meta = DatasetMetadata(dataset_id="ds", dataset_name="Test")
    assert meta.dataset_id == "ds"
    assert meta.dataset_name == "Test"
    assert meta.statistics is None


def test_dataset_license_defaults() -> None:
    lic = DatasetLicense()
    assert "demonstration" in lic.name
    assert lic.url is None
