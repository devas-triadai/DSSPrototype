"""Tests for the Trainer with MockTrainingBackend."""

import tempfile
from pathlib import Path

from backend.training.callbacks import Callback
from backend.training.checkpoint import CheckpointManager
from backend.training.evaluator import EvaluationEngine
from backend.training.experiment import ExperimentManager
from backend.training.exporter import ExportPipeline
from backend.training.history import HistoryManager
from backend.training.metrics import MetricsManager
from backend.training.models import (
    CheckpointData,
    ExportData,
    HistoryEntry,
    MetricData,
    TrainingConfigData,
)
from backend.training.registry import ModelRegistry
from backend.training.scheduler import Scheduler
from backend.training.trainer import Trainer

from .helpers import MockTrainingBackend


class _TestCallback(Callback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self) -> None:
        self.events.append("train_start")

    def on_epoch_start(self, epoch: int) -> None:
        self.events.append(f"epoch_start:{epoch}")

    def on_epoch_end(self, epoch: int, history_entry: HistoryEntry | None = None) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_validation_end(self, metrics: MetricData | None = None) -> None:
        self.events.append("validation_end")

    def on_checkpoint_saved(self, checkpoint: CheckpointData | None = None) -> None:
        self.events.append("checkpoint_saved")

    def on_export(self, export_data: ExportData | None = None) -> None:
        self.events.append("export")

    def on_train_end(self) -> None:
        self.events.append("train_end")


def _make_trainer(
    epochs: int = 3,
    callbacks: list[Callback] | None = None,
    scheduler: Scheduler | None = None,
) -> tuple[Trainer, Path]:
    tmp = Path(tempfile.mkdtemp())
    backend = MockTrainingBackend()
    trainer = Trainer(
        experiment_manager=ExperimentManager(experiments_dir=tmp / "experiments"),
        model_registry=ModelRegistry(models_dir=tmp / "models"),
        checkpoint_manager=CheckpointManager(checkpoints_dir=tmp / "checkpoints"),
        metrics_manager=MetricsManager(metrics_dir=tmp / "metrics"),
        history_manager=HistoryManager(history_dir=tmp / "history"),
        evaluation_engine=EvaluationEngine(reports_dir=tmp / "reports"),
        export_pipeline=ExportPipeline(exports_dir=tmp / "exports"),
        training_backend=backend,
        scheduler=scheduler or Scheduler(),
        callbacks=callbacks,
    )
    return trainer, tmp


def test_trainer_constructor() -> None:
    trainer, _ = _make_trainer()
    assert trainer is not None


def test_train_success() -> None:
    trainer, _ = _make_trainer(epochs=3)
    config = TrainingConfigData(
        model_name="test",
        epochs=3,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    assert result.status == "completed"
    assert result.total_epochs_completed == 3
    assert result.experiment_id.startswith("exp_")
    assert result.model_id.startswith("model_")


def test_train_creates_experiment() -> None:
    trainer, tmp = _make_trainer(epochs=2)
    config = TrainingConfigData(
        model_name="test",
        experiment_name="train_exp",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    exp_mgr = ExperimentManager(experiments_dir=tmp / "experiments")
    exp = exp_mgr.get_experiment(result.experiment_id)
    assert exp is not None
    assert exp.experiment_name == "train_exp"


def test_train_registers_model() -> None:
    trainer, tmp = _make_trainer(epochs=2)
    config = TrainingConfigData(
        model_name="test_model",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    registry = ModelRegistry(models_dir=tmp / "models")
    model = registry.get_model(result.model_id)
    assert model is not None
    assert model.model_name == "test_model"
    assert model.training_status == "completed"


def test_train_records_metrics() -> None:
    trainer, tmp = _make_trainer(epochs=3)
    config = TrainingConfigData(
        model_name="test",
        epochs=3,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    metrics_mgr = MetricsManager(metrics_dir=tmp / "metrics")
    metrics = metrics_mgr.get_metrics(result.experiment_id)
    assert len(metrics) >= 3


def test_train_records_history() -> None:
    trainer, tmp = _make_trainer(epochs=2)
    config = TrainingConfigData(
        model_name="test",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    hist_mgr = HistoryManager(history_dir=tmp / "history")
    history = hist_mgr.get_history(result.experiment_id)
    assert len(history) == 2


def test_train_creates_checkpoints() -> None:
    trainer, tmp = _make_trainer(epochs=3)
    config = TrainingConfigData(
        model_name="test",
        epochs=3,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    ckpt_mgr = CheckpointManager(checkpoints_dir=tmp / "checkpoints")
    checkpoints = ckpt_mgr.list_checkpoints(result.experiment_id)
    assert len(checkpoints) >= 3


def test_best_checkpoint_tracking() -> None:
    trainer, _ = _make_trainer(epochs=5)
    config = TrainingConfigData(
        model_name="test",
        epochs=5,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    assert result.best_metric is not None
    assert result.best_epoch is not None
    assert result.best_metric_name == "mAP50"


def test_training_duration_positive() -> None:
    trainer, _ = _make_trainer(epochs=2)
    config = TrainingConfigData(
        model_name="test",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    assert result.training_duration_seconds > 0


def test_callback_train_start_called() -> None:
    cb = _TestCallback()
    trainer, _ = _make_trainer(epochs=1, callbacks=[cb])
    config = TrainingConfigData(
        model_name="test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    trainer.train(config)
    assert "train_start" in cb.events


def test_callback_epoch_start_end_called() -> None:
    cb = _TestCallback()
    trainer, _ = _make_trainer(epochs=2, callbacks=[cb])
    config = TrainingConfigData(
        model_name="test",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    trainer.train(config)
    assert "epoch_start:1" in cb.events
    assert "epoch_start:2" in cb.events
    assert "epoch_end:1" in cb.events
    assert "epoch_end:2" in cb.events


def test_callback_validation_end_called() -> None:
    cb = _TestCallback()
    trainer, _ = _make_trainer(epochs=1, callbacks=[cb])
    config = TrainingConfigData(
        model_name="test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    trainer.train(config)
    assert "validation_end" in cb.events


def test_callback_checkpoint_saved_called() -> None:
    cb = _TestCallback()
    trainer, _ = _make_trainer(epochs=1, callbacks=[cb])
    config = TrainingConfigData(
        model_name="test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    trainer.train(config)
    assert "checkpoint_saved" in cb.events


def test_callback_train_end_called() -> None:
    cb = _TestCallback()
    trainer, _ = _make_trainer(epochs=1, callbacks=[cb])
    config = TrainingConfigData(
        model_name="test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    trainer.train(config)
    assert "train_end" in cb.events


def test_add_callback() -> None:
    trainer, _ = _make_trainer(epochs=1)
    cb = _TestCallback()
    trainer.add_callback(cb)
    config = TrainingConfigData(
        model_name="test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    trainer.train(config)
    assert "train_start" in cb.events


def test_final_metrics_returned() -> None:
    trainer, _ = _make_trainer(epochs=2)
    config = TrainingConfigData(
        model_name="test",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    assert result.final_metrics is not None
    assert result.final_metrics.epoch == 2


def test_final_status_completed() -> None:
    trainer, _ = _make_trainer(epochs=3)
    config = TrainingConfigData(
        model_name="test",
        epochs=3,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    assert result.status == "completed"


def test_experiment_status_transitions() -> None:
    trainer, tmp = _make_trainer(epochs=1)
    config = TrainingConfigData(
        model_name="test",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    result = trainer.train(config)
    exp_mgr = ExperimentManager(experiments_dir=tmp / "experiments")
    exp = exp_mgr.get_experiment(result.experiment_id)
    assert exp is not None
    assert exp.status == "completed"
