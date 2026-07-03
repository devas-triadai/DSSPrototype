"""Tests for the DatasetRegistry."""

from backend.dataset_manager.models import DatasetInfo
from backend.dataset_manager.registry import DatasetRegistry


def test_register_and_retrieve() -> None:
    registry = DatasetRegistry()
    info = DatasetInfo(dataset_id="test_001", dataset_name="test_dataset")
    registry.register(info)
    retrieved = registry.get("test_001")
    assert retrieved is not None
    assert retrieved.dataset_id == "test_001"
    assert retrieved.dataset_name == "test_dataset"


def test_get_by_name() -> None:
    registry = DatasetRegistry()
    info = DatasetInfo(dataset_id="test_002", dataset_name="by_name_test")
    registry.register(info)
    retrieved = registry.get_by_name("by_name_test")
    assert retrieved is not None
    assert retrieved.dataset_id == "test_002"


def test_list_datasets() -> None:
    registry = DatasetRegistry()
    registry.register(DatasetInfo(dataset_id="a", dataset_name="A"))
    registry.register(DatasetInfo(dataset_id="b", dataset_name="B"))
    assert len(registry.list_datasets()) == 2


def test_update() -> None:
    registry = DatasetRegistry()
    info = DatasetInfo(dataset_id="upd", dataset_name="original", image_count=0)
    registry.register(info)
    updated = DatasetInfo(
        dataset_id="upd", dataset_name="original", image_count=100,
    )
    registry.update(updated)
    retrieved = registry.get("upd")
    assert retrieved is not None
    assert retrieved.image_count == 100


def test_delete() -> None:
    registry = DatasetRegistry()
    registry.register(DatasetInfo(dataset_id="del", dataset_name="delete_me"))
    assert registry.contains("del") is True
    assert registry.delete("del") is True
    assert registry.contains("del") is False
    assert registry.delete("nonexistent") is False


def test_contains() -> None:
    registry = DatasetRegistry()
    registry.register(DatasetInfo(dataset_id="exists", dataset_name="exists"))
    assert registry.contains("exists") is True
    assert registry.contains("noexist") is False


def test_to_dict_from_dict_roundtrip() -> None:
    registry = DatasetRegistry()
    registry.register(DatasetInfo(dataset_id="r1", dataset_name="roundtrip_1"))
    registry.register(DatasetInfo(dataset_id="r2", dataset_name="roundtrip_2"))

    data = registry.to_dict()
    restored = DatasetRegistry.from_dict(data)
    assert restored.contains("r1") is True
    assert restored.contains("r2") is True
    assert len(restored.list_datasets()) == 2
