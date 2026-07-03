"""Tests for the DatasetSplitter."""

from pathlib import Path

from backend.dataset_manager.splitter import DatasetSplitter


def _make_paths(*names: str) -> list[Path]:
    return [Path(n) for n in names]


def test_split_default_ratios() -> None:
    images = _make_paths(*[f"img{i:03d}.jpg" for i in range(100)])
    split = DatasetSplitter().split(images, seed=42)
    assert len(split.train_images) == 70
    assert len(split.validation_images) == 15
    assert len(split.test_images) == 15


def test_split_custom_ratios() -> None:
    images = _make_paths(*[f"img{i:03d}.jpg" for i in range(100)])
    split = DatasetSplitter().split(
        images, train_ratio=0.8, validation_ratio=0.1, test_ratio=0.1, seed=42,
    )
    assert len(split.train_images) == 80
    assert len(split.validation_images) == 10
    assert len(split.test_images) == 10


def test_split_deterministic() -> None:
    images = _make_paths(*[f"img{i:03d}.jpg" for i in range(50)])
    s1 = DatasetSplitter().split(images, seed=123)
    s2 = DatasetSplitter().split(images, seed=123)
    assert s1.train_images == s2.train_images
    assert s1.validation_images == s2.validation_images
    assert s1.test_images == s2.test_images


def test_split_different_seeds_different() -> None:
    images = _make_paths(*[f"img{i:03d}.jpg" for i in range(50)])
    s1 = DatasetSplitter().split(images, seed=1)
    s2 = DatasetSplitter().split(images, seed=999)
    assert s1.train_images != s2.train_images


def test_split_with_annotations() -> None:
    img_paths = _make_paths("img001.jpg", "img002.jpg", "img003.jpg")
    ann_paths = _make_paths("img001.json", "img002.json")
    split = DatasetSplitter().split(img_paths, ann_paths, seed=42)
    assert split.train_annotations == split.train_annotations


def test_split_empty_images() -> None:
    split = DatasetSplitter().split([])
    assert split.train_images == []
    assert split.validation_images == []
    assert split.test_images == []


def test_split_single_image() -> None:
    images = _make_paths("img001.jpg")
    split = DatasetSplitter().split(images, seed=42)
    assert len(split.train_images) == 1
    assert len(split.validation_images) == 0
    assert len(split.test_images) == 0


def test_split_ratios_sum_to_one() -> None:
    images = _make_paths(*[f"img{i:03d}.jpg" for i in range(100)])
    split = DatasetSplitter().split(images, seed=42)
    total = len(split.train_images) + len(split.validation_images) + len(split.test_images)
    assert total == 100
