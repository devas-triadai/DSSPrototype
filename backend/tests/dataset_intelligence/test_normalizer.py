"""Tests for the DatasetNormalizer."""

from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    RawDataset,
)
from backend.dataset_intelligence.normalizer import DatasetNormalizer


def _make_raw_dataset() -> RawDataset:
    ann = Annotation(
        class_name="Main Battle Tank",
        bbox=(0.1, 0.2, 0.5, 0.6),
        bbox_format="xyxy_normalized",
    )
    img = ImageRecord(
        image_id="img001",
        image_path="/path/IMG_001.JPG",
        image_name="IMG_001.JPG",
        width=640,
        height=480,
        annotations=[ann],
    )
    return RawDataset(
        dataset_id="test_yolo",
        dataset_name="test",
        import_format="yolo",
        source_path="/path",
        images=[img],
        classes=["Main Battle Tank"],
    )


def test_normalize_class_name_snake_case() -> None:
    norm = DatasetNormalizer._normalize_class_name("Main Battle Tank")
    assert norm == "main_battle_tank"


def test_normalize_class_name_special_chars() -> None:
    norm = DatasetNormalizer._normalize_class_name("BMP-2/II")
    assert norm == "bmp2ii"


def test_normalize_class_name_whitespace() -> None:
    norm = DatasetNormalizer._normalize_class_name("  tank  ")
    assert norm == "tank"


def test_normalize_filename_lowercase() -> None:
    norm = DatasetNormalizer._normalize_filename("IMG_001.JPG")
    assert norm == "img_001.jpg"


def test_normalize_filename_special_chars() -> None:
    norm = DatasetNormalizer._normalize_filename("photo (1).PNG")
    assert norm == "photo-1.png"


def test_normalize_full_dataset() -> None:
    raw = _make_raw_dataset()
    normalizer = DatasetNormalizer()
    result = normalizer.normalize(raw)
    assert result.classes == ["main_battle_tank"]
    assert result.class_mapping == {"Main Battle Tank": "main_battle_tank"}
    assert len(result.images) == 1
    assert result.images[0].image_name == "img_001.jpg"
    assert result.images[0].annotations[0].normalized_class == "main_battle_tank"


def test_normalize_no_classes() -> None:
    raw = RawDataset(
        dataset_id="test",
        dataset_name="test",
        import_format="yolo",
        source_path="/path",
        images=[],
        classes=[],
    )
    normalizer = DatasetNormalizer()
    result = normalizer.normalize(raw)
    assert result.classes == []
    assert result.class_mapping == {}


def test_normalize_bbox_clamping() -> None:
    ann = Annotation(
        class_name="tank",
        bbox=(-0.1, 0.2, 1.5, 0.6),
        bbox_format="xyxy_normalized",
    )
    img = ImageRecord(
        image_id="img001",
        image_path="/path/img.jpg",
        image_name="img.jpg",
        width=640,
        height=480,
        annotations=[ann],
    )
    raw = RawDataset(
        dataset_id="test",
        dataset_name="test",
        import_format="yolo",
        source_path="/path",
        images=[img],
        classes=["tank"],
    )
    normalizer = DatasetNormalizer()
    result = normalizer.normalize(raw)
    x1, y1, x2, y2 = result.images[0].annotations[0].bbox
    assert x1 >= 0.0
    assert x2 <= 1.0
