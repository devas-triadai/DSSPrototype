"""Tests for the TrainingPipeline."""

import tempfile
from pathlib import Path

import pytest

from backend.training.augmentation import AugmentationPipeline
from backend.training.checkpoint import CheckpointManager
from backend.training.dataset_loader import DatasetLoader
from backend.training.evaluator import EvaluationEngine
from backend.training.exceptions import DatasetNotReadyError, PreValidationError
from backend.training.experiment import ExperimentManager
from backend.training.exporter import ExportPipeline
from backend.training.history import HistoryManager
from backend.training.hyperparameter_manager import HyperparameterManager
from backend.training.metrics import MetricsManager
from backend.training.models import AugmentationConfig, TrainingConfigData
from backend.training.registry import ModelRegistry
from backend.training.scheduler import Scheduler
from backend.training.training_pipeline import TrainingPipeline
from backend.training.validator import TrainingValidator

from .helpers import MockTrainingBackend


def _cleanup_dataset_yaml(dataset_version: str = "1.0") -> None:
    from pathlib import Path as _Path
    project_root = _Path(__file__).resolve().parent.parent.parent.parent
    yaml_dir = project_root / "datasets" / "exports" / "yolo" / dataset_version
    if yaml_dir.exists():
        import shutil
        shutil.rmtree(str(yaml_dir))


def _make_dataset_yaml(dataset_version: str = "1.0") -> Path:
    _cleanup_dataset_yaml(dataset_version)
    from pathlib import Path as _Path
    project_root = _Path(__file__).resolve().parent.parent.parent.parent
    yaml_dir = project_root / "datasets" / "exports" / "yolo" / dataset_version
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / "data.yaml"
    yaml_path.write_text("train: train\nval: val\nnc: 1\nnames: ['class1']")
    return yaml_path


def _make_pipeline() -> tuple[TrainingPipeline, Path, MockTrainingBackend]:
    tmp = Path(tempfile.mkdtemp())
    _make_dataset_yaml()
    backend = MockTrainingBackend()
    pipeline = TrainingPipeline(
        dataset_loader=DatasetLoader(reports_dir=tmp / "reports"),
        augmentation_pipeline=AugmentationPipeline(),
        hyperparameter_manager=HyperparameterManager(configs_dir=tmp / "configs"),
        validator=TrainingValidator(),
        experiment_manager=ExperimentManager(experiments_dir=tmp / "experiments"),
        model_registry=ModelRegistry(models_dir=tmp / "models"),
        checkpoint_manager=CheckpointManager(checkpoints_dir=tmp / "checkpoints"),
        metrics_manager=MetricsManager(metrics_dir=tmp / "metrics"),
        history_manager=HistoryManager(history_dir=tmp / "history"),
        evaluation_engine=EvaluationEngine(reports_dir=tmp / "eval_reports"),
        export_pipeline=ExportPipeline(exports_dir=tmp / "exports"),
        training_backend=backend,
        scheduler=Scheduler(),
    )
    return pipeline, tmp, backend


def _make_production_report(reports_dir: Path) -> None:
    import json
    report = {
        "dataset_name": "test_dataset",
        "dataset_version": "1.0",
        "production_ready": True,
        "overall_score": {"production_ready": True},
        "class_names": ["class1"],
        "num_classes": 1,
        "total_images": 100,
        "total_annotations": 500,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "test_dataset_1.0.json").write_text(json.dumps(report))


def test_pipeline_constructor() -> None:
    pipeline, _, _ = _make_pipeline()
    assert pipeline is not None


def test_pipeline_dataset_not_ready_error() -> None:
    pipeline, _, _ = _make_pipeline()
    config = TrainingConfigData(model_name="nonexistent", dataset_version="1.0")
    with pytest.raises(DatasetNotReadyError):
        pipeline.run(config)


def test_pipeline_validation_failure() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(model_name="test_dataset", dataset_version="1.0", batch_size=0)
    with pytest.raises(PreValidationError):
        pipeline.run(config)


def test_pipeline_successful_run() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=2,
        save_interval=1,
        validation_interval=1,
        experiment_name="pipe_test",
    )
    result = pipeline.run(config)
    assert result.experiment_id is not None
    assert result.model_id is not None
    assert result.status == "completed"
    assert result.total_epochs_completed == 2


def test_pipeline_with_augmentation_config() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    aug_cfg = AugmentationConfig(name="custom_aug", flip_probability=0.0)
    result = pipeline.run(config, augmentation_config=aug_cfg)
    assert result.status == "completed"


def test_pipeline_creates_experiment() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=1,
        save_interval=1,
        validation_interval=1,
        experiment_name="exp_created",
    )
    result = pipeline.run(config)
    exp_mgr = ExperimentManager(experiments_dir=tmp / "experiments")
    exp = exp_mgr.get_experiment(result.experiment_id)
    assert exp is not None
    assert exp.experiment_name == "exp_created"


def test_pipeline_registers_model() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=1,
        save_interval=1,
        validation_interval=1,
    )
    result = pipeline.run(config)
    registry = ModelRegistry(models_dir=tmp / "models")
    model = registry.get_model(result.model_id)
    assert model is not None
    assert model.model_name == "test_dataset"


def test_pipeline_records_metrics() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = pipeline.run(config)
    metrics_mgr = MetricsManager(metrics_dir=tmp / "metrics")
    metrics = metrics_mgr.get_metrics(result.experiment_id)
    assert len(metrics) >= 2


def test_pipeline_records_history() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = pipeline.run(config)
    hist_mgr = HistoryManager(history_dir=tmp / "history")
    history = hist_mgr.get_history(result.experiment_id)
    assert len(history) == 2


def test_pipeline_creates_checkpoints() -> None:
    pipeline, tmp, _ = _make_pipeline()
    _make_production_report(tmp / "reports")
    config = TrainingConfigData(
        model_name="test_dataset",
        dataset_version="1.0",
        epochs=2,
        save_interval=1,
        validation_interval=1,
    )
    result = pipeline.run(config)
    ckpt_mgr = CheckpointManager(checkpoints_dir=tmp / "checkpoints")
    checkpoints = ckpt_mgr.list_checkpoints(result.experiment_id)
    assert len(checkpoints) >= 2
