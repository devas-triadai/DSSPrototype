"""Tests for the StatisticsEngine."""

from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
)
from backend.dataset_intelligence.statistics import StatisticsEngine


def _make_dataset(
    num_images: int = 10,
    anns_per_image: int = 3,
    classes: list[str] | None = None,
) -> NormalizedDataset:
    if classes is None:
        classes = ["tank"]
    images = []
    for i in range(num_images):
        anns = []
        for j in range(anns_per_image):
            cls = classes[j % len(classes)]
            ann = Annotation(
                class_name=cls,
                normalized_class=cls,
                ontology_class=cls,
                bbox=(0.1, 0.2, 0.5, 0.6),
            )
            anns.append(ann)
        img = ImageRecord(
            image_id=f"img{i:03d}",
            image_path=f"/path/img{i:03d}.jpg",
            image_name=f"img{i:03d}.jpg",
            width=640 + i,
            height=480 + i,
            annotations=anns,
        )
        images.append(img)
    return NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        images=images,
        classes=classes,
    )


def test_statistics_basic_counts() -> None:
    dataset = _make_dataset(10, 3, ["tank"])
    engine = StatisticsEngine()
    stats = engine.compute(dataset)
    assert stats.total_images == 10
    assert stats.total_annotations == 30
    assert stats.class_count == 1


def test_statistics_class_distribution() -> None:
    dataset = _make_dataset(10, 4, ["tank", "truck", "jeep"])
    engine = StatisticsEngine()
    stats = engine.compute(dataset)
    assert stats.class_count == 3
    assert len(stats.objects_per_class) == 3


def test_statistics_average_objects() -> None:
    dataset = _make_dataset(10, 5, ["tank"])
    engine = StatisticsEngine()
    stats = engine.compute(dataset)
    assert stats.average_objects_per_image == 5.0


def test_statistics_imbalance_ratio() -> None:
    dataset = _make_dataset(10, 1, ["tank", "truck"])
    engine = StatisticsEngine()
    stats = engine.compute(dataset)
    assert stats.class_imbalance_ratio >= 0


def test_statistics_empty_dataset() -> None:
    engine = StatisticsEngine()
    stats = engine.compute(
        NormalizedDataset(dataset_id="empty", dataset_name="empty")
    )
    assert stats.total_images == 0
    assert stats.total_annotations == 0
    assert stats.class_count == 0


def test_statistics_diversity_score() -> None:
    dataset = _make_dataset(10, 4, ["tank", "truck"])
    engine = StatisticsEngine()
    stats = engine.compute(dataset)
    assert 0.0 <= stats.dataset_diversity <= 1.0


def test_statistics_resolution_distribution() -> None:
    dataset = _make_dataset(5, 1, ["tank"])
    engine = StatisticsEngine()
    stats = engine.compute(dataset)
    assert stats.average_image_width > 0
    assert stats.average_image_height > 0
