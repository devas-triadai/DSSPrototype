"""Tests for the DuplicateDetector."""

from backend.dataset_intelligence.duplicate_detector import DuplicateDetector
from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
)


def _make_dataset(images: list[ImageRecord]) -> NormalizedDataset:
    classes = sorted(
        {ann.class_name for img in images for ann in img.annotations}
    )
    return NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        images=images,
        classes=classes,
    )


def _make_image(
    image_id: str,
    name: str,
    checksum: str = "",
    ann_class: str = "tank",
) -> ImageRecord:
    ann = Annotation(class_name=ann_class, bbox=(0.1, 0.2, 0.5, 0.6))
    return ImageRecord(
        image_id=image_id,
        image_path=f"/path/{name}",
        image_name=name,
        width=640,
        height=480,
        annotations=[ann],
        checksum=checksum,
    )


def test_no_duplicates() -> None:
    ann1 = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.3, 0.4))
    ann2 = Annotation(class_name="truck", bbox=(0.5, 0.6, 0.7, 0.8))
    img1 = ImageRecord(
        image_id="img001", image_path="/p/img001.jpg", image_name="img001.jpg",
        width=640, height=480, annotations=[ann1], checksum="hash1",
        metadata={"source": "a"},
    )
    img2 = ImageRecord(
        image_id="img002", image_path="/p/img002.jpg", image_name="img002.jpg",
        width=640, height=480, annotations=[ann2], checksum="hash2",
        metadata={"source": "b"},
    )
    detector = DuplicateDetector()
    report = detector.detect(_make_dataset([img1, img2]))
    assert len(report.duplicates) == 0
    assert report.duplicate_ratio == 0.0


def test_filename_duplicates() -> None:
    images = [
        _make_image("img001", "same.jpg", "hash1"),
        _make_image("img002", "same.jpg", "hash2"),
    ]
    detector = DuplicateDetector()
    report = detector.detect(_make_dataset(images))
    assert any(d.duplicate_type == "filename" for d in report.duplicates)


def test_hash_duplicates() -> None:
    images = [
        _make_image("img001", "img001.jpg", "same_hash"),
        _make_image("img002", "img002.jpg", "same_hash"),
    ]
    detector = DuplicateDetector()
    report = detector.detect(_make_dataset(images))
    assert any(d.duplicate_type == "hash" for d in report.duplicates)


def test_annotation_duplicates() -> None:
    ann1 = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    ann2 = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
    img = ImageRecord(
        image_id="img001",
        image_path="/path/img.jpg",
        image_name="img.jpg",
        width=640,
        height=480,
        annotations=[ann1, ann2],
        checksum="hash1",
    )
    detector = DuplicateDetector()
    report = detector.detect(_make_dataset([img]))
    assert any(d.duplicate_type == "annotation" for d in report.duplicates)


def test_object_duplicates_across_images() -> None:
    images = [
        _make_image("img001", "img001.jpg", "hash1", "tank"),
        _make_image("img002", "img002.jpg", "hash2", "tank"),
    ]
    detector = DuplicateDetector()
    report = detector.detect(_make_dataset(images))
    # Same class + same bbox across images
    object_dups = [d for d in report.duplicates if d.duplicate_type == "object"]
    assert len(object_dups) > 0


def test_empty_dataset() -> None:
    detector = DuplicateDetector()
    report = detector.detect(
        NormalizedDataset(dataset_id="empty", dataset_name="empty")
    )
    assert len(report.duplicates) == 0
    assert report.duplicate_ratio == 0.0
