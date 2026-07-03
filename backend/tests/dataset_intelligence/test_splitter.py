"""Tests for the DatasetSplitter."""

from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
)
from backend.dataset_intelligence.splitter import DatasetSplitter


def _make_dataset(num_images: int = 100) -> NormalizedDataset:
    images = []
    for i in range(num_images):
        ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
        img = ImageRecord(
            image_id=f"img{i:03d}",
            image_path=f"/path/img{i:03d}.jpg",
            image_name=f"img{i:03d}.jpg",
            width=640,
            height=480,
            annotations=[ann],
        )
        images.append(img)
    return NormalizedDataset(
        dataset_id="test",
        dataset_name="test",
        images=images,
        classes=["tank"],
    )


def test_split_default_ratios() -> None:
    dataset = _make_dataset(100)
    splitter = DatasetSplitter()
    splits = splitter.split(dataset)
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits
    total = len(splits["train"]) + len(splits["validation"]) + len(splits["test"])
    assert total == 100


def test_split_custom_ratios() -> None:
    dataset = _make_dataset(100)
    splitter = DatasetSplitter()
    splits = splitter.split(dataset, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1)
    assert len(splits["train"]) >= 80
    assert len(splits["validation"]) >= 8
    assert len(splits["test"]) >= 8


def test_split_invalid_ratios() -> None:
    dataset = _make_dataset(100)
    splitter = DatasetSplitter()
    try:
        splitter.split(dataset, train_ratio=0.5, validation_ratio=0.5, test_ratio=0.5)
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_split_empty_dataset() -> None:
    dataset = NormalizedDataset(dataset_id="empty", dataset_name="empty")
    splitter = DatasetSplitter()
    splits = splitter.split(dataset)
    assert splits == {"train": [], "validation": [], "test": []}


def test_split_seed_determinism() -> None:
    dataset = _make_dataset(100)
    splitter = DatasetSplitter()
    splits1 = splitter.split(dataset, seed=42)
    splits2 = splitter.split(dataset, seed=42)
    assert splits1["train"] == splits2["train"]


def test_split_stratified_vs_random() -> None:
    dataset = _make_dataset(100)
    splitter = DatasetSplitter()
    stratified = splitter.split(dataset, stratified=True)
    random = splitter.split(dataset, stratified=False)
    assert "train" in stratified
    assert "train" in random
