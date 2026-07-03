"""Tests for SourceRegistry."""

import tempfile
from pathlib import Path

from backend.dataset_catalog.exceptions import SourceNotFoundError
from backend.dataset_catalog.models import SourceInfo
from backend.dataset_catalog.source_registry import SourceRegistry


def _make_source(source_id: str = "src_001") -> SourceInfo:
    return SourceInfo(
        source_id=source_id,
        name="Test Source",
        source_type="url",
        url="https://example.com/data",
        description="A test source",
    )


def test_register_and_get() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        src = _make_source()
        reg.register_source(src)
        assert reg.get_source("src_001") == src


def test_get_nonexistent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        assert reg.get_source("ghost") is None


def test_update_source() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        reg.register_source(_make_source("src_001"))
        updated = SourceInfo(
            source_id="src_001",
            name="Updated Source",
            source_type="api",
            reliability=0.9,
        )
        reg.update_source(updated)
        src = reg.get_source("src_001")
        assert src is not None
        assert src.name == "Updated Source"
        assert src.reliability == 0.9


def test_update_nonexistent_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        try:
            reg.update_source(_make_source("ghost"))
            assert False, "Expected SourceNotFoundError"
        except SourceNotFoundError:
            pass


def test_list_sources() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        reg.register_source(_make_source("src_001"))
        reg.register_source(
            SourceInfo(source_id="src_002", name="API Source", source_type="api")
        )
        reg.register_source(
            SourceInfo(source_id="src_003", name="Local Source", source_type="local")
        )
        assert len(reg.list_sources()) == 3
        assert len(reg.list_sources(source_type="api")) == 1
        assert len(reg.list_sources(source_type="url")) == 1


def test_record_success() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        reg.register_source(_make_source("src_001"))
        reg.record_success("src_001")
        src = reg.get_source("src_001")
        assert src is not None
        assert src.successful_fetches == 1
        assert src.total_fetches == 1
        assert src.reliability == 1.0


def test_record_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        reg.register_source(_make_source("src_001"))
        reg.record_failure("src_001", "Connection timeout")
        src = reg.get_source("src_001")
        assert src is not None
        assert src.failed_fetches == 1
        assert src.total_fetches == 1
        assert "timeout" in src.last_error


def test_record_success_nonexistent_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        try:
            reg.record_success("ghost")
            assert False, "Expected SourceNotFoundError"
        except SourceNotFoundError:
            pass


def test_record_failure_nonexistent_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        try:
            reg.record_failure("ghost", "error")
            assert False, "Expected SourceNotFoundError"
        except SourceNotFoundError:
            pass


def test_get_reliability() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        reg.register_source(_make_source("src_001"))
        assert reg.get_reliability("src_001") == 0.5
        reg.record_success("src_001")
        reg.record_success("src_001")
        reg.record_success("src_001")
        assert reg.get_reliability("src_001") == 1.0


def test_get_reliability_nonexistent_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        reg = SourceRegistry(Path(tmp) / "sources.json")
        try:
            reg.get_reliability("ghost")
            assert False, "Expected SourceNotFoundError"
        except SourceNotFoundError:
            pass


def test_persistence() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "sources.json"
        reg1 = SourceRegistry(db)
        reg1.register_source(_make_source("src_001"))
        reg1.register_source(_make_source("src_002"))
        reg2 = SourceRegistry(db)
        assert len(reg2.list_sources()) == 2
