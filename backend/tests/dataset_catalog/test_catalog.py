"""Tests for the JSON-backed Catalog store."""

import tempfile
from pathlib import Path

from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.exceptions import CatalogError, EntryNotFoundError
from backend.dataset_catalog.models import CatalogEntry


def _make_entry(entry_id: str = "e_001", name: str = "Test") -> CatalogEntry:
    return CatalogEntry(
        entry_id=entry_id,
        name=name,
        source_id="src_001",
        source_type="local",
        domain="military",
        status="profiled",
        tags=["test"],
    )


def test_add_and_get_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.json"
        cat = Catalog(db)
        entry = _make_entry()
        cat.add_entry(entry)
        assert cat.get_entry("e_001") == entry


def test_add_duplicate_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        cat.add_entry(_make_entry())
        try:
            cat.add_entry(_make_entry())
            assert False, "Expected CatalogError"
        except CatalogError:
            pass


def test_get_nonexistent_returns_none() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        assert cat.get_entry("nonexistent") is None


def test_update_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        cat.add_entry(_make_entry("e_001", "Original"))
        updated = _make_entry("e_001", "Updated")
        cat.update_entry(updated)
        entry = cat.get_entry("e_001")
        assert entry is not None
        assert entry.name == "Updated"


def test_update_nonexistent_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        try:
            cat.update_entry(_make_entry("ghost"))
            assert False, "Expected EntryNotFoundError"
        except EntryNotFoundError:
            pass


def test_remove_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        cat.add_entry(_make_entry())
        assert cat.remove_entry("e_001") is True
        assert cat.get_entry("e_001") is None


def test_remove_nonexistent_returns_false() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        assert cat.remove_entry("ghost") is False


def test_list_entries_all() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        cat.add_entry(_make_entry("e_001"))
        cat.add_entry(_make_entry("e_002"))
        assert len(cat.list_entries()) == 2


def test_list_entries_filtered() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        a = _make_entry("e_001", name="A")
        b = CatalogEntry(
            entry_id="e_002",
            name="B",
            source_id="src_002",
            source_type="url",
            status="acquired",
        )
        cat.add_entry(a)
        cat.add_entry(b)
        assert len(cat.list_entries(status="profiled")) == 1
        assert len(cat.list_entries(source_type="url")) == 1
        assert len(cat.list_entries(domain="air")) == 0


def test_search_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        cat.add_entry(
            CatalogEntry(
                entry_id="e_001",
                name="Tank Dataset",
                source_id="src_001",
                source_type="local",
                tags=["armor", "ground"],
            )
        )
        cat.add_entry(
            CatalogEntry(
                entry_id="e_002",
                name="Aircraft Dataset",
                source_id="src_002",
                source_type="url",
                tags=["air", "fighter"],
            )
        )
        assert len(cat.search_entries("tank")) == 1
        assert len(cat.search_entries("fighter")) == 1
        assert len(cat.search_entries("dataset")) == 2


def test_count_entries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat = Catalog(Path(tmp) / "catalog.json")
        assert cat.count_entries() == 0
        cat.add_entry(_make_entry("e_001"))
        assert cat.count_entries() == 1


def test_persistence_across_reload() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.json"
        cat1 = Catalog(db)
        cat1.add_entry(_make_entry("e_001"))
        cat1.add_entry(_make_entry("e_002"))
        cat2 = Catalog(db)
        assert cat2.count_entries() == 2
        entry = cat2.get_entry("e_001")
        assert entry is not None
        assert entry.name == "Test"


def test_load_corrupted_file_starts_fresh() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "catalog.json"
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_text("{invalid json", encoding="utf-8")
        cat = Catalog(db)
        assert cat.count_entries() == 0
