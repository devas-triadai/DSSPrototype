"""Tests for the DatasetMerger."""

from backend.dataset_intelligence.merger import DatasetMerger
from backend.dataset_intelligence.models import (
    Annotation,
    ImageRecord,
    NormalizedDataset,
)


def _make_dataset(dataset_id: str, name: str, num_images: int = 2) -> NormalizedDataset:
    images = []
    for i in range(num_images):
        ann = Annotation(class_name="tank", bbox=(0.1, 0.2, 0.5, 0.6))
        img = ImageRecord(
            image_id=f"{dataset_id}_img{i:03d}",
            image_path=f"/path/{dataset_id}_img{i:03d}.jpg",
            image_name=f"{dataset_id}_img{i:03d}.jpg",
            width=640,
            height=480,
            annotations=[ann],
        )
        images.append(img)
    return NormalizedDataset(
        dataset_id=dataset_id,
        dataset_name=name,
        images=images,
        classes=["tank"],
    )


def test_merge_two_datasets() -> None:
    ds1 = _make_dataset("ds1", "dataset_a", 2)
    ds2 = _make_dataset("ds2", "dataset_b", 3)
    merger = DatasetMerger()
    result = merger.merge([ds1, ds2])
    assert len(result.images) == 5
    assert "ds1" in result.source_datasets
    assert "ds2" in result.source_datasets


def test_merge_with_class_overlap() -> None:
    ds1 = _make_dataset("ds1", "dataset_a")
    ds2 = _make_dataset("ds2", "dataset_b")
    merger = DatasetMerger()
    result = merger.merge([ds1, ds2])
    assert "tank" in result.classes


def test_merge_empty_list() -> None:
    merger = DatasetMerger()
    try:
        merger.merge([])
        assert False, "Should raise ValueError"
    except ValueError:
        assert True


def test_merge_single_dataset() -> None:
    ds = _make_dataset("ds1", "dataset_a", 2)
    merger = DatasetMerger()
    result = merger.merge([ds])
    assert len(result.images) == 2
