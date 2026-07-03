"""Tests for CurationService."""

import tempfile
from pathlib import Path

from backend.dataset_catalog.catalog import Catalog
from backend.dataset_catalog.curation_service import CurationService
from backend.dataset_catalog.exceptions import (
    CurationError,
    CurationWorkflowError,
    EntryNotFoundError,
)
from backend.dataset_catalog.models import CatalogEntry


def _setup(tmp: Path) -> tuple[Catalog, CurationService]:
    cat = Catalog(tmp / "catalog.json")
    curation = CurationService(cat, work_dir=tmp / "work")
    entry = CatalogEntry(
        entry_id="e_001",
        name="Test Dataset",
        source_id="src_001",
        source_type="local",
    )
    cat.add_entry(entry)
    return cat, curation


def test_create_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        assert record.entry_id == "e_001"
        assert record.curator == "analyst_01"
        assert record.status == "draft"


def test_create_record_nonexistent_entry() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        try:
            curation.create_record("ghost", "analyst_01")
            assert False, "Expected EntryNotFoundError"
        except EntryNotFoundError:
            pass


def test_submit_for_review() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        submitted = curation.submit_for_review(record.record_id)
        assert submitted.status == "pending_review"


def test_submit_non_draft_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        curation.submit_for_review(record.record_id)
        try:
            curation.submit_for_review(record.record_id)
            assert False, "Expected CurationWorkflowError"
        except CurationWorkflowError:
            pass


def test_submit_nonexistent_record_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        try:
            curation.submit_for_review("ghost")
            assert False, "Expected CurationError"
        except CurationError:
            pass


def test_approve() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cat, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        curation.submit_for_review(record.record_id)
        approved = curation.approve(record.record_id, "manager_01")
        assert approved.status == "approved"
        assert approved.reviewer == "manager_01"
        # Catalog entry status should be updated
        entry = cat.get_entry("e_001")
        assert entry is not None
        assert entry.status == "acquired"


def test_approve_non_pending_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        try:
            curation.approve(record.record_id, "manager_01")
            assert False, "Expected CurationWorkflowError"
        except CurationWorkflowError:
            pass


def test_reject() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        curation.submit_for_review(record.record_id)
        rejected = curation.reject(record.record_id, "manager_01", "Insufficient quality")
        assert rejected.status == "rejected"
        assert rejected.rejection_reason == "Insufficient quality"


def test_reject_non_pending_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        try:
            curation.reject(record.record_id, "manager_01", "Bad")
            assert False, "Expected CurationWorkflowError"
        except CurationWorkflowError:
            pass


def test_get_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        record = curation.create_record("e_001", "analyst_01")
        loaded = curation.get_record(record.record_id)
        assert loaded is not None
        assert loaded.record_id == record.record_id


def test_get_nonexistent_record() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        assert curation.get_record("ghost") is None


def test_list_pending() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        r1 = curation.create_record("e_001", "analyst_01")
        curation.submit_for_review(r1.record_id)
        curation.create_record("e_001", "analyst_02")
        pending = curation.list_pending()
        assert len(pending) == 1


def test_list_by_curator() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        _, curation = _setup(Path(tmp))
        curation.create_record("e_001", "analyst_01")
        curation.create_record("e_001", "analyst_02")
        assert len(curation.list_by_curator("analyst_01")) == 1
        assert len(curation.list_by_curator("analyst_02")) == 1


def test_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cat = Catalog(tmp_path / "catalog.json")
        cat.add_entry(
            CatalogEntry(entry_id="e_001", name="Test", source_id="src_001", source_type="local")
        )
        work_dir = tmp_path / "work"

        cs1 = CurationService(cat, work_dir=work_dir)
        cs1.create_record("e_001", "analyst_01")

        cs2 = CurationService(cat, work_dir=work_dir)
        records = cs2.list_by_curator("analyst_01")
        assert len(records) == 1
