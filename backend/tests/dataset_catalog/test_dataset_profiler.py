"""Tests for DatasetProfiler."""

from pathlib import Path

from backend.dataset_catalog.dataset_profile import DatasetProfiler
from backend.dataset_catalog.exceptions import ProfileError
from backend.dataset_catalog.models import DatasetProfile


def test_profile_nonexistent_path_raises() -> None:
    profiler = DatasetProfiler()
    try:
        profiler.profile(
            Path("/nonexistent/path"),
            source_id="src_001",
            source_type="local",
        )
        assert False, "Expected ProfileError"
    except ProfileError:
        pass


def test_profile_empty_directory_raises(tmp_path: Path) -> None:
    profiler = DatasetProfiler()
    try:
        profiler.profile(
            tmp_path / "empty",
            source_id="src_001",
            source_type="local",
        )
        assert False, "Expected ProfileError"
    except ProfileError:
        pass


def test_profile_with_images(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    img_dir = dataset_dir / "images"
    img_dir.mkdir()
    jpeg_header = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07"
    )
    (img_dir / "img_001.jpg").write_bytes(jpeg_header)
    (img_dir / "img_002.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

    profiler = DatasetProfiler()
    profile = profiler.profile(dataset_dir, source_id="src_001", source_type="local")

    assert isinstance(profile, DatasetProfile)
    assert profile.source_id == "src_001"
    assert profile.total_images == 2
    assert profile.profile_id == "src_001_dataset"


def test_profile_with_yolo_annotations(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "yolo_dataset"
    dataset_dir.mkdir()
    img_dir = dataset_dir / "images"
    img_dir.mkdir()
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir()
    jpeg_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01"
    (img_dir / "img_001.jpg").write_bytes(jpeg_data)
    (img_dir / "img_002.jpg").write_bytes(jpeg_data)
    (labels_dir / "img_001.txt").write_text("0 0.5 0.5 0.2 0.3\n1 0.3 0.4 0.1 0.2\n")
    (labels_dir / "img_002.txt").write_text("0 0.6 0.6 0.3 0.4\n")

    profiler = DatasetProfiler()
    profile = profiler.profile(dataset_dir, source_id="src_001", source_type="local")

    assert profile.total_images == 2
    assert profile.annotation_format == "yolo"
    assert profile.total_annotations >= 3


def test_get_profile(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    profiler = DatasetProfiler(profiles_dir=profiles_dir)

    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir()
    (dataset_dir / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    profile = profiler.profile(dataset_dir, source_id="src_001", source_type="local")
    loaded = profiler.get_profile(profile.profile_id)
    assert loaded is not None
    assert loaded.source_id == "src_001"


def test_get_nonexistent_profile(tmp_path: Path) -> None:
    profiler = DatasetProfiler(profiles_dir=tmp_path / "profiles")
    assert profiler.get_profile("ghost") is None


def test_update_profile() -> None:
    profiler = DatasetProfiler()
    profile = DatasetProfile(
        profile_id="prof_001",
        source_id="src_001",
        source_type="local",
        path="/data",
        tags=["test"],
    )
    profiler.update_profile(profile)
    loaded = profiler.get_profile("prof_001")
    assert loaded is not None
    assert loaded.source_id == "src_001"
    assert loaded.tags == ["test"]


def test_compare_profiles() -> None:
    profiler = DatasetProfiler()
    p1 = DatasetProfile(
        profile_id="p1", source_id="s1", source_type="local", path="/a",
        classes=["tank", "apc"], total_images=10,
    )
    p2 = DatasetProfile(
        profile_id="p2", source_id="s2", source_type="local", path="/b",
        classes=["tank", "helicopter"], total_images=20,
    )
    profiler.update_profile(p1)
    profiler.update_profile(p2)

    result = profiler.compare_profiles(["p1", "p2"])
    assert result["profile_count"] == 2
    common = result["common_classes"]
    assert isinstance(common, list)
    assert "tank" in common


def test_detect_license(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "licensed_ds"
    dataset_dir.mkdir()
    (dataset_dir / "LICENSE").write_text("Creative Commons Attribution 4.0", encoding="utf-8")
    (dataset_dir / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")

    profiler = DatasetProfiler()
    profile = profiler.profile(dataset_dir, source_id="src_001", source_type="local")
    assert profile.license_info is not None
    assert profile.license_info.license_id == "cc_by_4"
