"""Tests for the DatasetManagementService."""

import tempfile
from pathlib import Path

from backend.dataset_manager.models import DatasetInfo
from backend.dataset_manager.registry import DatasetRegistry
from backend.dataset_manager.service import DatasetManagementService
from backend.dataset_manager.versioning import DatasetVersioning


def _make_service() -> DatasetManagementService:
    """Create a service with isolated versioning for clean test state."""
    tmp = Path(tempfile.mkdtemp())
    versioning = DatasetVersioning(versions_dir=tmp / "versions")
    return DatasetManagementService(versioning=versioning)


def _create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake_image_data")


def _create_annotation(path: Path, content: str = '{"annotations": []}') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_service_register_and_list() -> None:
    service = DatasetManagementService()
    info = DatasetInfo(dataset_id="svc_001", dataset_name="SVC Test")
    service.register_dataset(info)
    datasets = service.list_datasets()
    assert len(datasets) == 1
    assert datasets[0].dataset_id == "svc_001"


def test_service_get_dataset() -> None:
    service = DatasetManagementService()
    info = DatasetInfo(dataset_id="get_test", dataset_name="Get Test")
    service.register_dataset(info)
    retrieved = service.get_dataset("get_test")
    assert retrieved is not None
    assert retrieved.dataset_name == "Get Test"


def test_service_get_nonexistent() -> None:
    service = DatasetManagementService()
    assert service.get_dataset("nonexistent") is None


def test_service_delete_dataset() -> None:
    service = DatasetManagementService()
    service.register_dataset(DatasetInfo(dataset_id="del_me", dataset_name="Del Me"))
    assert service.delete_dataset("del_me") is True
    assert service.get_dataset("del_me") is None


def test_service_create_version() -> None:
    service = _make_service()
    service.register_dataset(DatasetInfo(dataset_id="ver_test", dataset_name="Ver Test"))
    version = service.create_version("ver_test", change_log="Initial")
    assert version is not None
    assert version.version == "1.0.0"
    assert version.change_log == "Initial"


def test_service_list_versions() -> None:
    service = _make_service()
    service.register_dataset(DatasetInfo(dataset_id="ver_list", dataset_name="Ver List"))
    service.create_version("ver_list")
    service.create_version("ver_list", change_log="v2")
    versions = service.list_versions("ver_list")
    assert len(versions) == 2


def test_service_split_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        raw_dir = Path(tmp) / "raw"
        ann_dir = Path(tmp) / "annotated"
        raw_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        for i in range(10):
            _create_image(raw_dir / f"img{i:03d}.jpg")

        from backend.dataset_manager.config import dm_config
        original_raw = dm_config.raw_dir
        original_ann = dm_config.annotated_dir
        try:
            dm_config.raw_dir = raw_dir
            dm_config.annotated_dir = ann_dir

            service = DatasetManagementService()
            service.register_dataset(
                DatasetInfo(dataset_id="split_test", dataset_name="Split Test"),
            )

            split = service.split_dataset("split_test", seed=42)
            assert len(split.train_images) > 0
            total = len(split.train_images) + len(split.validation_images)
            total += len(split.test_images)
            assert total == 10
        finally:
            dm_config.raw_dir = original_raw
            dm_config.annotated_dir = original_ann


def test_service_export_dataset() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        raw_dir = tmp_path / "raw"
        ann_dir = tmp_path / "annotated"
        raw_dir.mkdir(parents=True)
        ann_dir.mkdir(parents=True)
        _create_image(raw_dir / "img001.jpg")

        from backend.dataset_manager.config import dm_config
        original_raw = dm_config.raw_dir
        original_ann = dm_config.annotated_dir
        try:
            dm_config.raw_dir = raw_dir
            dm_config.annotated_dir = ann_dir

            versioning = DatasetVersioning(versions_dir=tmp_path / "versions")
            service = DatasetManagementService(versioning=versioning)
            service.register_dataset(DatasetInfo(
                dataset_id="exp_test",
                dataset_name="Export Test",
                classes=["person"],
            ))
            out_dir = Path(tmp) / "yolo_exp"
            export = service.export_dataset(
                "exp_test", format_name="yolo", output_dir=out_dir,
            )
            assert export.format_name == "yolo"
            assert export.image_count > 0
        finally:
            dm_config.raw_dir = original_raw
            dm_config.annotated_dir = original_ann


def test_service_custom_registry() -> None:
    custom_registry = DatasetRegistry()
    service = DatasetManagementService(registry=custom_registry)
    info = DatasetInfo(dataset_id="custom", dataset_name="Custom Registry")
    service.register_dataset(info)
    assert custom_registry.contains("custom") is True
    assert service.get_dataset("custom") is not None
