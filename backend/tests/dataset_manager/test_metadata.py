"""Tests for the MetadataGenerator."""

from pathlib import Path

from backend.dataset_manager.metadata import MetadataGenerator
from backend.dataset_manager.models import DatasetInfo, DatasetMetadata


def test_generate_minimal() -> None:
    info = DatasetInfo(dataset_id="minimal", dataset_name="Minimal Dataset")
    meta = MetadataGenerator().generate(dataset_info=info)
    assert isinstance(meta, DatasetMetadata)
    assert meta.dataset_id == "minimal"
    assert meta.dataset_name == "Minimal Dataset"
    assert meta.statistics is None
    assert meta.quality is None
    assert meta.validation is None


def test_generate_with_all_fields() -> None:
    from backend.dataset_manager.models import (
        DatasetChecksum,
        DatasetQuality,
        DatasetStatistics,
        DatasetValidation,
    )

    info = DatasetInfo(
        dataset_id="full",
        dataset_name="Full Dataset",
        image_count=50,
        annotation_count=200,
        class_count=3,
        classes=["cat", "dog", "bird"],
    )
    stats = DatasetStatistics(dataset_id="full", total_images=50, total_annotations=200)
    quality = DatasetQuality(dataset_id="full", quality_score=0.95)
    validation = DatasetValidation(
        dataset_id="full", passed=True, total_checks=12,
        passed_checks=12, failed_checks=0,
    )
    cs = DatasetChecksum(algorithm="sha256", value="abc123", file_path="/tmp/test")

    meta = MetadataGenerator().generate(
        dataset_info=info,
        statistics=stats,
        quality=quality,
        validation=validation,
        checksum=cs,
    )
    assert meta.statistics is not None
    assert meta.statistics.total_images == 50
    assert meta.quality is not None
    assert meta.quality.quality_score == 0.95
    assert meta.validation is not None
    assert meta.validation.passed is True
    assert meta.checksum is not None
    assert meta.checksum.value == "abc123"


def test_persists_metadata_file() -> None:
    import tempfile as tf

    gen = MetadataGenerator()
    original_dir = gen._config.metadata_dir
    try:
        tmp = Path(tf.mkdtemp())
        gen._config.metadata_dir = tmp
        info = DatasetInfo(dataset_id="persist_test", dataset_name="Persist")
        gen.generate(dataset_info=info)
        files = list(tmp.glob("*"))
        assert len(files) >= 1
        assert files[0].name == "persist_test_metadata.json"
    finally:
        gen._config.metadata_dir = original_dir
