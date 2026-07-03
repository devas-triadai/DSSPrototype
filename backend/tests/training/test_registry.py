"""Tests for the ModelRegistry."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.training.exceptions import ModelNotFoundError
from backend.training.models import ModelEntry
from backend.training.registry import ModelRegistry


def _make_registry() -> tuple[ModelRegistry, Path]:
    tmp = Path(tempfile.mkdtemp())
    return ModelRegistry(models_dir=tmp / "models"), tmp


def test_register_and_get() -> None:
    reg, _ = _make_registry()
    entry = ModelEntry(model_id="m_001", model_name="yolo")
    reg.register_model(entry)
    retrieved = reg.get_model("m_001")
    assert retrieved is not None
    assert retrieved.model_name == "yolo"


def test_get_nonexistent() -> None:
    reg, _ = _make_registry()
    assert reg.get_model("nonexistent") is None


def test_get_models_by_name() -> None:
    reg, _ = _make_registry()
    reg.register_model(ModelEntry(model_id="m_001", model_name="yolo", version="1.0.0"))
    reg.register_model(ModelEntry(model_id="m_002", model_name="yolo", version="2.0.0"))
    reg.register_model(ModelEntry(model_id="m_003", model_name="resnet", version="1.0.0"))
    yolo_models = reg.get_models_by_name("yolo")
    assert len(yolo_models) == 2


def test_get_models_by_name_empty() -> None:
    reg, _ = _make_registry()
    assert reg.get_models_by_name("nonexistent") == []


def test_list_models() -> None:
    reg, _ = _make_registry()
    reg.register_model(ModelEntry(model_id="a", model_name="A"))
    reg.register_model(ModelEntry(model_id="b", model_name="B"))
    assert len(reg.list_models()) == 2


def test_list_models_empty() -> None:
    reg, _ = _make_registry()
    assert reg.list_models() == []


def test_update_model() -> None:
    reg, _ = _make_registry()
    entry = ModelEntry(model_id="upd", model_name="original")
    reg.register_model(entry)
    updated = ModelEntry(
        model_id="upd", model_name="original",
        training_status="completed", metrics={"mAP50": 0.85},
    )
    reg.update_model(updated)
    retrieved = reg.get_model("upd")
    assert retrieved is not None
    assert retrieved.training_status == "completed"
    assert retrieved.metrics["mAP50"] == 0.85


def test_update_model_not_found_raises() -> None:
    reg, _ = _make_registry()
    entry = ModelEntry(model_id="nonexistent", model_name="test")
    with pytest.raises(ModelNotFoundError):
        reg.update_model(entry)


def test_delete_model() -> None:
    reg, _ = _make_registry()
    reg.register_model(ModelEntry(model_id="del", model_name="delete"))
    assert reg.delete_model("del") is True
    assert reg.get_model("del") is None


def test_delete_nonexistent_model() -> None:
    reg, _ = _make_registry()
    assert reg.delete_model("nonexistent") is False


def test_register_persists_to_disk() -> None:
    reg, tmp = _make_registry()
    entry = ModelEntry(model_id="persist", model_name="yolo")
    reg.register_model(entry)
    model_file = tmp / "models" / "persist.json"
    assert model_file.exists()
    data = json.loads(model_file.read_text())
    assert data["model_name"] == "yolo"


def test_get_model_from_disk_cache() -> None:
    reg, tmp = _make_registry()
    entry = ModelEntry(model_id="disk_cache", model_name="yolo", version="2.0.0")
    reg.register_model(entry)
    reg2 = ModelRegistry(models_dir=tmp / "models")
    retrieved = reg2.get_model("disk_cache")
    assert retrieved is not None
    assert retrieved.version == "2.0.0"


def test_update_model_persists_metrics() -> None:
    reg, _ = _make_registry()
    entry = ModelEntry(model_id="metrics_test", model_name="yolo")
    reg.register_model(entry)
    updated = ModelEntry(
        model_id="metrics_test", model_name="yolo",
        metrics={"mAP50": 0.9, "mAP50_95": 0.7},
    )
    reg.update_model(updated)
    retrieved = reg.get_model("metrics_test")
    assert retrieved is not None
    assert retrieved.metrics["mAP50_95"] == 0.7
