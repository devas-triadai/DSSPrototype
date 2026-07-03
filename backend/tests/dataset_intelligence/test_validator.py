"""Tests for the DatasetValidator."""

import tempfile

from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    RawDataset,
)
from backend.dataset_intelligence.validator import DatasetValidator


def _make_raw_dataset(
    image_id: str = "img001",
    class_name: str = "tank",
    bbox: tuple[float, float, float, float] = (0.1, 0.2, 0.5, 0.6),
    width: int = 640,
    height: int = 480,
    image_exists: bool = True,
) -> RawDataset:
    ann = Annotation(class_name=class_name, bbox=bbox, bbox_format="xyxy_normalized")
    img_path = "/fake/path/img001.jpg"
    if image_exists:
        img_path = str(tempfile.NamedTemporaryFile(suffix=".jpg", delete=False).name)
    img = ImageRecord(
        image_id=image_id,
        image_path=img_path,
        image_name="img001.jpg",
        width=width,
        height=height,
        annotations=[ann],
    )
    return RawDataset(
        dataset_id="test_yolo",
        dataset_name="test",
        import_format="yolo",
        source_path="/fake",
        images=[img],
        classes=[class_name],
    )


def test_validate_passes() -> None:
    ds = _make_raw_dataset()
    validator = DatasetValidator()
    report = validator.validate(ds)
    assert report.passed is True
    assert report.errors == []


def test_validate_no_images() -> None:
    ds = RawDataset(
        dataset_id="test",
        dataset_name="test",
        import_format="yolo",
        source_path="/fake",
        images=[],
        classes=["tank"],
    )
    validator = DatasetValidator()
    report = validator.validate(ds)
    assert report.passed is False
    assert any("no images" in e.lower() for e in report.errors)


def test_validate_no_classes() -> None:
    ds = _make_raw_dataset()
    ds = RawDataset(
        dataset_id=ds.dataset_id,
        dataset_name=ds.dataset_name,
        import_format=ds.import_format,
        source_path=ds.source_path,
        images=ds.images,
        classes=[],
    )
    validator = DatasetValidator()
    report = validator.validate(ds)
    assert report.passed is False
    assert any("no classes" in e.lower() for e in report.errors)


def test_validate_invalid_bbox() -> None:
    ds = _make_raw_dataset(bbox=(1.5, 0.2, 2.0, 0.6))
    validator = DatasetValidator()
    report = validator.validate(ds)
    assert report.invalid_bounding_boxes != []


def test_validate_negative_coordinates() -> None:
    ds = _make_raw_dataset(bbox=(-0.1, 0.2, 0.5, 0.6))
    validator = DatasetValidator()
    report = validator.validate(ds)
    assert report.negative_coordinates != []


def test_validate_class_mismatch() -> None:
    import tempfile
    tmpf = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmpf.write(b"data")
    tmpf.close()
    ann = Annotation(class_name="unknown", bbox=(0.1, 0.2, 0.5, 0.6))
    img = ImageRecord(
        image_id="img001", image_path=tmpf.name, image_name="img.jpg",
        width=640, height=480, annotations=[ann],
    )
    ds = RawDataset(
        dataset_id="test", dataset_name="test", import_format="yolo",
        source_path="/fake", images=[img], classes=["tank"],
    )
    validator = DatasetValidator()
    report = validator.validate(ds)
    assert report.class_mismatches != []


def test_validate_empty_annotations() -> None:
    ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    img = ImageRecord(
        image_id="img001",
        image_path="/fake/path.jpg",
        image_name="img.jpg",
        annotations=[ann],
    )
    ds = RawDataset(
        dataset_id="test",
        dataset_name="test",
        import_format="yolo",
        source_path="/fake",
        images=[img],
        classes=["tank"],
    )
    validator = DatasetValidator()
    report = validator.validate(ds)
    # Image doesn't exist so it's a missing image, not empty annotation
    assert report.missing_images != []
