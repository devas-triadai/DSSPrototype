"""Test fixtures for dataset intelligence tests."""

import tempfile
from pathlib import Path

import pytest

from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
    ProvenanceRecord,
    RawDataset,
)


@pytest.fixture
def sample_annotation() -> Annotation:
    return Annotation(
        class_name="tank",
        bbox=(0.1, 0.2, 0.5, 0.6),
        bbox_format="xyxy_normalized",
    )


@pytest.fixture
def sample_image_record(sample_annotation: Annotation) -> ImageRecord:
    return ImageRecord(
        image_id="img001",
        image_path=str(Path("/fake/path/img001.jpg")),
        image_name="img001.jpg",
        width=640,
        height=480,
        format="jpg",
        annotations=[sample_annotation],
        checksum="abc123",
        provenance=ProvenanceRecord(
            source_dataset="test_dataset",
            original_class="tank",
            import_format="yolo",
        ),
    )


@pytest.fixture
def sample_raw_dataset(sample_image_record: ImageRecord) -> RawDataset:
    return RawDataset(
        dataset_id="test_dataset_yolo",
        dataset_name="test_dataset",
        import_format="yolo",
        source_path=str(Path("/fake/source")),
        images=[sample_image_record],
        classes=["tank"],
    )


@pytest.fixture
def sample_normalized_dataset(sample_image_record: ImageRecord) -> NormalizedDataset:
    return NormalizedDataset(
        dataset_id="test_dataset_yolo",
        dataset_name="test_dataset",
        images=[sample_image_record],
        classes=["main_battle_tank"],
        class_mapping={"tank": "main_battle_tank"},
        normalization_log=["Class remap: tank -> main_battle_tank"],
    )


@pytest.fixture
def temp_dir() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        yield tmp


@pytest.fixture
def temp_image_path(temp_dir: str) -> Path:
    img_path = Path(temp_dir) / "test_image.jpg"
    img_path.write_bytes(b"fake_image_bytes")
    return img_path
