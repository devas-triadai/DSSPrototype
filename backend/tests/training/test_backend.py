"""Tests for the training backend system: interface, registry, mock backend."""

import tempfile
from pathlib import Path

from backend.training.backends.registry import (
    TrainingBackendRegistry,
    get_default_registry,
)
from backend.training.models import TrainingConfigData

from .helpers import MockTrainingBackend


def test_mock_backend_initialize() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    assert backend._initialized


def test_mock_backend_train_epoch() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    metrics = backend.train_epoch(1, 0.001)
    assert "training_loss" in metrics
    assert metrics["learning_rate"] == 0.001
    assert metrics["training_loss"] > 0


def test_mock_backend_validate() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    backend.train_epoch(1, 0.001)
    result = backend.validate("/fake/ckpt.pt", "/fake/dataset", "exp_001")
    assert result.mAP50 is not None
    assert result.mAP50 > 0


def test_mock_backend_test() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    result = backend.test("/fake/ckpt.pt", "/fake/dataset", "exp_001")
    assert result.split == "test"
    assert result.mAP50 is not None


def test_mock_backend_export() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    tmp = Path(tempfile.mkdtemp())
    path = backend.export("/fake/ckpt.pt", "onnx", tmp, "m_001", "exp_001")
    assert Path(path).exists()


def test_mock_backend_save_checkpoint() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    tmp = Path(tempfile.mkdtemp()) / "ckpt.pt"
    meta = backend.save_checkpoint(str(tmp), 5)
    assert meta["epoch"] == 5
    assert Path(str(meta["path"])).exists()


def test_mock_backend_shutdown() -> None:
    backend = MockTrainingBackend()
    cfg = TrainingConfigData(model_name="test", epochs=2)
    backend.initialize(cfg, "exp_001", "/fake/dataset")
    backend.shutdown()
    assert not backend._initialized


def test_registry_register_and_get() -> None:
    registry = TrainingBackendRegistry()
    registry.register("mock", MockTrainingBackend)
    cls = registry.get("mock")
    assert cls is MockTrainingBackend


def test_registry_create() -> None:
    registry = TrainingBackendRegistry()
    registry.register("mock", MockTrainingBackend)
    backend = registry.create("mock")
    assert isinstance(backend, MockTrainingBackend)


def test_registry_list() -> None:
    registry = TrainingBackendRegistry()
    registry.register("mock", MockTrainingBackend)
    backends = registry.list_backends()
    assert "mock" in backends


def test_registry_get_unknown_raises() -> None:
    registry = TrainingBackendRegistry()
    import pytest
    with pytest.raises(KeyError, match="Unknown training backend"):
        registry.get("nonexistent")


def test_default_registry_is_singleton() -> None:
    assert get_default_registry() is get_default_registry()
