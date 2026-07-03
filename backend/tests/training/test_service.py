"""Tests for the TrainingService."""

import tempfile
from pathlib import Path

from backend.training.models import (
    ModelEntry,
    TrainingConfigData,
)
from backend.training.service import TrainingService

from .helpers import MockTrainingBackend


def _make_service() -> TrainingService:
    tmp = Path(tempfile.mkdtemp())
    from backend.training.checkpoint import CheckpointManager
    from backend.training.evaluator import EvaluationEngine
    from backend.training.experiment import ExperimentManager
    from backend.training.exporter import ExportPipeline
    from backend.training.history import HistoryManager
    from backend.training.metrics import MetricsManager
    from backend.training.registry import ModelRegistry
    return TrainingService(
        experiment_manager=ExperimentManager(experiments_dir=tmp / "experiments"),
        model_registry=ModelRegistry(models_dir=tmp / "models"),
        checkpoint_manager=CheckpointManager(checkpoints_dir=tmp / "checkpoints"),
        metrics_manager=MetricsManager(metrics_dir=tmp / "metrics"),
        history_manager=HistoryManager(history_dir=tmp / "history"),
        evaluation_engine=EvaluationEngine(reports_dir=tmp / "reports"),
        export_pipeline=ExportPipeline(exports_dir=tmp / "exports"),
        training_backend=MockTrainingBackend(),
    )


def test_service_create_experiment() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(model_name="test", experiment_name="svc_exp")
    exp = svc.create_experiment(cfg)
    assert exp.experiment_name == "svc_exp"
    assert exp.status == "created"


def test_service_get_experiment() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(model_name="test", experiment_name="get_test")
    exp = svc.create_experiment(cfg)
    retrieved = svc.get_experiment(exp.experiment_id)
    assert retrieved is not None


def test_service_list_experiments() -> None:
    svc = _make_service()
    svc.create_experiment(TrainingConfigData(model_name="a", experiment_name="exp_a"))
    svc.create_experiment(TrainingConfigData(model_name="b", experiment_name="exp_b"))
    assert len(svc.list_experiments()) == 2


def test_service_register_model() -> None:
    svc = _make_service()
    entry = ModelEntry(model_id="svc_m_001", model_name="yolo")
    svc.register_model(entry)
    retrieved = svc.get_model("svc_m_001")
    assert retrieved is not None


def test_service_list_models() -> None:
    svc = _make_service()
    svc.register_model(ModelEntry(model_id="m1", model_name="yolo"))
    svc.register_model(ModelEntry(model_id="m2", model_name="resnet"))
    assert len(svc.list_models()) == 2


def test_service_train_executes() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(
        model_name="yolo",
        experiment_name="train_test",
        epochs=3,
        save_interval=1,
        validation_interval=1,
    )
    result = svc.train(cfg)
    assert result.experiment_id is not None
    assert result.model_id is not None
    assert result.total_epochs_completed == 3
    assert result.status == "completed"
    assert result.training_duration_seconds > 0


def test_service_get_metrics() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(
        model_name="test",
        experiment_name="metrics_test",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = svc.train(cfg)
    exp_id = result.experiment_id
    metrics = svc.get_metrics(exp_id)
    assert len(metrics) >= 2


def test_service_get_history() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(
        model_name="test",
        experiment_name="history_test",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = svc.train(cfg)
    history = svc.get_history(result.experiment_id)
    assert len(history) == 2


def test_service_get_best_checkpoint() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(
        model_name="test",
        experiment_name="ckpt_test",
        epochs=3,
        save_interval=1,
        validation_interval=1,
    )
    result = svc.train(cfg)
    best = svc.get_best_checkpoint(result.experiment_id)
    assert best is not None


def test_service_export_model() -> None:
    svc = _make_service()
    cfg = TrainingConfigData(
        model_name="test",
        experiment_name="export_test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    result = svc.train(cfg)
    tmp = Path(tempfile.mkdtemp())
    exports = svc.export_model(
        result.experiment_id, result.model_id, tmp, formats=["onnx", "torchscript"],
    )
    assert len(exports) == 2
    assert exports[0].format_name == "onnx"
    assert exports[1].format_name == "torchscript"


def test_service_no_backend_raises() -> None:
    svc = TrainingService()
    cfg = TrainingConfigData(model_name="test", experiment_name="no_backend")
    import pytest
    with pytest.raises(RuntimeError, match="No training backend configured"):
        svc.train(cfg)
