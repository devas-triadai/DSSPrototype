"""Tests for the DatasetIntelligenceRegistry."""

import tempfile
from pathlib import Path

from backend.dataset_intelligence.exceptions import DatasetNotFoundError
from backend.dataset_intelligence.models import DatasetIntelligenceRegistryEntry
from backend.dataset_intelligence.registry import DatasetIntelligenceRegistry


def _make_entry(
    dataset_id: str = "test_ds", name: str = "test"
) -> DatasetIntelligenceRegistryEntry:
    return DatasetIntelligenceRegistryEntry(
        dataset_id=dataset_id,
        dataset_name=name,
    )


def test_register_and_get() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        entry = _make_entry()
        reg.register(entry)
        retrieved = reg.get("test_ds")
        assert retrieved is not None
        assert retrieved.dataset_name == "test"


def test_get_nonexistent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        assert reg.get("nonexistent") is None


def test_list_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        reg.register(_make_entry("ds1", "first"))
        reg.register(_make_entry("ds2", "second"))
        entries = reg.list_entries()
        assert len(entries) == 2


def test_update_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        entry = _make_entry("ds1", "original")
        reg.register(entry)
        updated = DatasetIntelligenceRegistryEntry(
            dataset_id="ds1",
            dataset_name="updated",
            status="ready",
            quality_score=0.95,
        )
        reg.update(updated)
        retrieved = reg.get("ds1")
        assert retrieved is not None
        assert retrieved.dataset_name == "updated"
        assert retrieved.quality_score == 0.95


def test_update_nonexistent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        try:
            reg.update(_make_entry("nonexistent"))
            assert False
        except DatasetNotFoundError:
            assert True


def test_delete_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        reg.register(_make_entry("ds1"))
        assert reg.delete("ds1") is True
        assert reg.get("ds1") is None


def test_delete_nonexistent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = DatasetIntelligenceRegistry(Path(tmp) / "registry.json")
        assert reg.delete("nonexistent") is False


def test_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg_path = Path(tmp) / "registry.json"
        reg1 = DatasetIntelligenceRegistry(reg_path)
        reg1.register(_make_entry("ds1", "persistent_test"))
        reg2 = DatasetIntelligenceRegistry(reg_path)
        retrieved = reg2.get("ds1")
        assert retrieved is not None
        assert retrieved.dataset_name == "persistent_test"
