"""Tests for the DatasetVersioning."""

import tempfile
from pathlib import Path

from backend.dataset_manager.versioning import DatasetVersioning


def _versioning_with_temp_dir() -> tuple[DatasetVersioning, Path]:
    """Create a DatasetVersioning instance backed by a temp directory."""
    tmp = Path(tempfile.mkdtemp())
    v = DatasetVersioning()
    # Override the versions dir to point to temp
    v._versions_dir = tmp
    return v, tmp


def test_create_first_version() -> None:
    v, _ = _versioning_with_temp_dir()
    ver = v.create_version("ds_001")
    assert ver.version == "1.0.0"
    assert ver.dataset_id == "ds_001"


def test_create_version_with_custom_version() -> None:
    v, _ = _versioning_with_temp_dir()
    ver = v.create_version("ds_002", version="2.0.0", change_log="Initial release")
    assert ver.version == "2.0.0"
    assert ver.change_log == "Initial release"


def test_auto_increment() -> None:
    v, _ = _versioning_with_temp_dir()
    v.create_version("ds_003")
    v2 = v.create_version("ds_003", change_log="Bug fix")
    assert v2.version == "1.0.1"


def test_get_version() -> None:
    v, _ = _versioning_with_temp_dir()
    v.create_version("ds_004", version="1.5.0")
    retrieved = v.get_version("ds_004", "1.5.0")
    assert retrieved is not None
    assert retrieved.version == "1.5.0"


def test_get_version_not_found() -> None:
    v, _ = _versioning_with_temp_dir()
    assert v.get_version("nonexistent", "1.0.0") is None


def test_list_versions() -> None:
    v, _ = _versioning_with_temp_dir()
    v.create_version("ds_005")
    v.create_version("ds_005", change_log="v2")
    v.create_version("ds_005", change_log="v3")
    versions = v.list_versions("ds_005")
    assert len(versions) == 3
    assert versions[0].version >= versions[1].version


def test_latest_version() -> None:
    v, _ = _versioning_with_temp_dir()
    v.create_version("ds_006")
    v.create_version("ds_006", change_log="latest")
    latest = v.get_latest_version("ds_006")
    assert latest is not None
    assert latest.change_log == "latest"


def test_bump_major_version() -> None:
    v, _ = _versioning_with_temp_dir()
    v.create_version("ds_007", version="9.9.19")
    v2 = v.create_version("ds_007")
    assert v2.version == "10.0.0"


def test_multiple_datasets_independent_versions() -> None:
    v, _ = _versioning_with_temp_dir()
    v.create_version("ds_a")
    v.create_version("ds_b")
    v.create_version("ds_a", change_log="A v2")
    assert len(v.list_versions("ds_a")) == 2
    assert len(v.list_versions("ds_b")) == 1
