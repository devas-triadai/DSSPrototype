"""Tests for all 16 training Pydantic models."""

import pytest

from backend.training.models import (
    AugmentationConfig,
    CheckpointData,
    DatasetLoadResult,
    EarlyStoppingConfig,
    EvaluationResult,
    ExperimentData,
    ExportData,
    HistoryEntry,
    HyperparameterProfile,
    MetricData,
    ModelEntry,
    PreValidationResult,
    SchedulerConfig,
    TrainingConfigData,
    TrainingResult,
)

# ── TrainingConfigData ────────────────────────────────────────────────

def test_training_config_data_defaults() -> None:
    cfg = TrainingConfigData(model_name="yolo")
    assert cfg.model_name == "yolo"
    assert cfg.epochs == 100
    assert cfg.batch_size == 16
    assert cfg.learning_rate == 0.001
    assert cfg.optimizer == "adam"
    assert cfg.scheduler == "cosine"
    assert cfg.weight_decay == 0.0001
    assert cfg.image_size == (640, 640)
    assert cfg.augmentation == "default"
    assert cfg.mixed_precision is False
    assert cfg.device == "cpu"
    assert cfg.workers == 4
    assert cfg.seed == 42
    assert cfg.resume_checkpoint is None
    assert cfg.early_stopping_patience is None
    assert cfg.early_stopping_delta == 0.001
    assert cfg.validation_interval == 1
    assert cfg.save_interval == 5
    assert cfg.notes == ""
    assert cfg.model_version == "1.0.0"
    assert cfg.dataset_version == "1.0.0"
    assert cfg.experiment_name == "default"


def test_training_config_validation_passes() -> None:
    cfg = TrainingConfigData(model_name="test")
    errors = cfg.validate_config()
    assert errors == []


def test_training_config_validation_batch_size() -> None:
    cfg = TrainingConfigData(model_name="test", batch_size=0)
    errors = cfg.validate_config()
    assert "batch_size must be >= 1" in errors


def test_training_config_validation_epochs() -> None:
    cfg = TrainingConfigData(model_name="test", epochs=0)
    errors = cfg.validate_config()
    assert "epochs must be >= 1" in errors


def test_training_config_validation_learning_rate() -> None:
    cfg = TrainingConfigData(model_name="test", learning_rate=0)
    errors = cfg.validate_config()
    assert "learning_rate must be positive" in errors


def test_training_config_validation_learning_rate_negative() -> None:
    cfg = TrainingConfigData(model_name="test", learning_rate=-0.1)
    errors = cfg.validate_config()
    assert "learning_rate must be positive" in errors


def test_training_config_validation_validation_interval() -> None:
    cfg = TrainingConfigData(model_name="test", validation_interval=0)
    errors = cfg.validate_config()
    assert "validation_interval must be >= 1" in errors


def test_training_config_validation_save_interval() -> None:
    cfg = TrainingConfigData(model_name="test", save_interval=0)
    errors = cfg.validate_config()
    assert "save_interval must be >= 1" in errors


def test_training_config_validation_multiple_errors() -> None:
    cfg = TrainingConfigData(model_name="test", batch_size=0, epochs=0, learning_rate=0)
    errors = cfg.validate_config()
    assert len(errors) >= 3


def test_training_config_frozen() -> None:
    cfg = TrainingConfigData(model_name="test")
    with pytest.raises(Exception):
        cfg.model_name = "changed"


def test_training_config_serialization_roundtrip() -> None:
    cfg = TrainingConfigData(model_name="yolo", epochs=50, batch_size=8)
    data = cfg.model_dump()
    restored = TrainingConfigData(**data)
    assert restored.model_name == "yolo"
    assert restored.epochs == 50
    assert restored.batch_size == 8


# ── ExperimentData ────────────────────────────────────────────────────

def test_experiment_data_defaults() -> None:
    cfg = TrainingConfigData(model_name="test")
    exp = ExperimentData(experiment_id="exp_001", experiment_name="test", config=cfg)
    assert exp.status == "created"
    assert exp.best_epoch is None
    assert exp.training_end is None
    assert exp.dataset_version == ""
    assert exp.duration_seconds is None
    assert exp.best_metric is None
    assert exp.best_metric_name == ""
    assert exp.notes == ""


def test_experiment_data_frozen() -> None:
    cfg = TrainingConfigData(model_name="test")
    exp = ExperimentData(experiment_id="exp_002", experiment_name="test", config=cfg)
    with pytest.raises(Exception):
        exp.status = "running"


def test_experiment_data_training_start_is_isoformat() -> None:
    cfg = TrainingConfigData(model_name="test")
    exp = ExperimentData(experiment_id="exp_003", experiment_name="test", config=cfg)
    assert "T" in exp.training_start


def test_experiment_data_serialization_roundtrip() -> None:
    cfg = TrainingConfigData(model_name="test")
    exp = ExperimentData(
        experiment_id="exp_004", experiment_name="test", config=cfg,
        status="completed", best_epoch=5, best_metric=0.95,
    )
    data = exp.model_dump()
    restored = ExperimentData(**data)
    assert restored.experiment_id == "exp_004"
    assert restored.status == "completed"
    assert restored.best_epoch == 5
    assert restored.best_metric == 0.95


# ── ModelEntry ────────────────────────────────────────────────────────

def test_model_entry_defaults() -> None:
    m = ModelEntry(model_id="m_001", model_name="yolo")
    assert m.version == "1.0.0"
    assert m.training_status == "pending"
    assert m.framework == "pytorch"
    assert m.metrics == {}
    assert m.architecture == ""
    assert m.dataset_version == ""
    assert m.experiment_id == ""
    assert m.checkpoint_path == ""
    assert m.export_formats == []
    assert m.checksum == ""


def test_model_entry_frozen() -> None:
    m = ModelEntry(model_id="m_002", model_name="yolo")
    with pytest.raises(Exception):
        m.model_name = "resnet"


def test_model_entry_with_metrics() -> None:
    m = ModelEntry(
        model_id="m_003", model_name="yolo",
        metrics={"mAP50": 0.85, "mAP50_95": 0.65},
    )
    assert m.metrics["mAP50"] == 0.85


def test_model_entry_serialization_roundtrip() -> None:
    m = ModelEntry(
        model_id="m_004", model_name="yolo", version="2.0.0",
        training_status="completed", framework="pytorch",
    )
    data = m.model_dump()
    restored = ModelEntry(**data)
    assert restored.model_id == "m_004"
    assert restored.training_status == "completed"


# ── CheckpointData ────────────────────────────────────────────────────

def test_checkpoint_data_defaults() -> None:
    c = CheckpointData(experiment_id="exp_001", epoch=5, path="/tmp/ckpt.pt")
    assert c.is_best is False
    assert c.is_latest is False
    assert c.metric_value is None
    assert c.file_size_bytes == 0
    assert c.checksum == ""
    assert c.metadata == {}


def test_checkpoint_data_frozen() -> None:
    c = CheckpointData(experiment_id="exp_001", epoch=1, path="/tmp/ckpt.pt")
    with pytest.raises(Exception):
        c.is_best = True


def test_checkpoint_data_with_metric() -> None:
    c = CheckpointData(
        experiment_id="exp_001", epoch=5, path="/tmp/ckpt.pt",
        metric_value=0.85, is_best=True, is_latest=True,
    )
    assert c.metric_value == 0.85
    assert c.is_best is True
    assert c.is_latest is True


def test_checkpoint_data_serialization_roundtrip() -> None:
    c = CheckpointData(
        experiment_id="exp_001", epoch=3, path="/tmp/ckpt.pt",
        metric_value=0.9, is_best=True, is_latest=True,
    )
    data = c.model_dump()
    restored = CheckpointData(**data)
    assert restored.epoch == 3
    assert restored.is_best is True


# ── MetricData ────────────────────────────────────────────────────────

def test_metric_data_defaults() -> None:
    m = MetricData(experiment_id="exp_001", epoch=1)
    assert m.training_loss is None
    assert m.validation_loss is None
    assert m.additional_metrics == {}
    assert m.step == 0
    assert m.precision is None
    assert m.recall is None
    assert m.mAP50 is None
    assert m.mAP50_95 is None
    assert m.learning_rate is None
    assert m.epoch_time_ms is None
    assert m.gpu_memory_mb is None
    assert m.cpu_memory_mb is None


def test_metric_data_frozen() -> None:
    m = MetricData(experiment_id="exp_001", epoch=1)
    with pytest.raises(Exception):
        m.training_loss = 0.5


def test_metric_data_with_values() -> None:
    m = MetricData(
        experiment_id="exp_001", epoch=1, training_loss=0.5,
        validation_loss=0.4, mAP50=0.85, learning_rate=0.001,
    )
    assert m.training_loss == 0.5
    assert m.validation_loss == 0.4
    assert m.mAP50 == 0.85
    assert m.learning_rate == 0.001


def test_metric_data_with_additional() -> None:
    m = MetricData(
        experiment_id="exp_001", epoch=1, training_loss=0.5,
        additional_metrics={"box_loss": 0.3, "cls_loss": 0.2},
    )
    assert m.additional_metrics["box_loss"] == 0.3


def test_metric_data_serialization_roundtrip() -> None:
    m = MetricData(experiment_id="exp_001", epoch=2, training_loss=0.3, mAP50=0.9)
    data = m.model_dump()
    restored = MetricData(**data)
    assert restored.epoch == 2
    assert restored.mAP50 == 0.9


# ── HistoryEntry ──────────────────────────────────────────────────────

def test_history_entry_defaults() -> None:
    h = HistoryEntry(experiment_id="exp_001", epoch=1, loss=0.5, learning_rate=0.001)
    assert h.metrics == {}
    assert h.checkpoint_path == ""


def test_history_entry_frozen() -> None:
    h = HistoryEntry(experiment_id="exp_001", epoch=1, loss=0.5, learning_rate=0.001)
    with pytest.raises(Exception):
        h.loss = 0.3


def test_history_entry_with_metrics() -> None:
    h = HistoryEntry(
        experiment_id="exp_001", epoch=1, loss=0.5,
        learning_rate=0.001, metrics={"mAP50": 0.85},
        checkpoint_path="/tmp/ckpt.pt",
    )
    assert h.metrics["mAP50"] == 0.85
    assert h.checkpoint_path == "/tmp/ckpt.pt"


def test_history_entry_serialization_roundtrip() -> None:
    h = HistoryEntry(experiment_id="exp_001", epoch=3, loss=0.2, learning_rate=0.0001)
    data = h.model_dump()
    restored = HistoryEntry(**data)
    assert restored.epoch == 3
    assert restored.loss == 0.2


# ── EvaluationResult ──────────────────────────────────────────────────

def test_evaluation_result_defaults() -> None:
    e = EvaluationResult(experiment_id="exp_001", checkpoint_path="/tmp/ckpt.pt")
    assert e.split == "validation"
    assert e.per_class_metrics == {}
    assert e.dataset_version == ""
    assert e.precision is None
    assert e.recall is None
    assert e.mAP50 is None
    assert e.mAP50_95 is None
    assert e.classification_accuracy is None
    assert e.inference_time_ms is None
    assert e.total_images == 0


def test_evaluation_result_frozen() -> None:
    e = EvaluationResult(experiment_id="exp_001", checkpoint_path="/tmp/ckpt.pt")
    with pytest.raises(Exception):
        e.split = "test"


def test_evaluation_result_with_metrics() -> None:
    e = EvaluationResult(
        experiment_id="exp_001", checkpoint_path="/tmp/ckpt.pt",
        split="test", mAP50=0.85, mAP50_95=0.65, total_images=100,
    )
    assert e.mAP50 == 0.85
    assert e.split == "test"
    assert e.total_images == 100


def test_evaluation_result_per_class() -> None:
    e = EvaluationResult(
        experiment_id="exp_001", checkpoint_path="/tmp/ckpt.pt",
        per_class_metrics={"person": {"mAP": 0.9}},
    )
    assert e.per_class_metrics["person"]["mAP"] == 0.9


def test_evaluation_result_serialization_roundtrip() -> None:
    e = EvaluationResult(
        experiment_id="exp_001", checkpoint_path="/tmp/ckpt.pt",
        mAP50=0.85, split="test",
    )
    data = e.model_dump()
    restored = EvaluationResult(**data)
    assert restored.mAP50 == 0.85
    assert restored.split == "test"


# ── ExportData ────────────────────────────────────────────────────────

def test_export_data_defaults() -> None:
    e = ExportData(
        experiment_id="exp_001", model_id="m_001", format_name="onnx",
        output_path="/tmp/model.onnx",
    )
    assert e.opset_version is None
    assert e.file_size_bytes == 0
    assert e.checksum == ""
    assert e.framework_version == ""
    assert e.input_shape is None
    assert e.output_shape is None


def test_export_data_frozen() -> None:
    e = ExportData(
        experiment_id="exp_001", model_id="m_001", format_name="onnx",
        output_path="/tmp/model.onnx",
    )
    with pytest.raises(Exception):
        e.format_name = "torchscript"


def test_export_data_with_opset() -> None:
    e = ExportData(
        experiment_id="exp_001", model_id="m_001", format_name="onnx",
        output_path="/tmp/model.onnx", opset_version=17,
    )
    assert e.opset_version == 17


def test_export_data_serialization_roundtrip() -> None:
    e = ExportData(
        experiment_id="exp_001", model_id="m_001", format_name="onnx",
        output_path="/tmp/model.onnx", file_size_bytes=1024,
    )
    data = e.model_dump()
    restored = ExportData(**data)
    assert restored.format_name == "onnx"
    assert restored.file_size_bytes == 1024


# ── SchedulerConfig ───────────────────────────────────────────────────

def test_scheduler_config_defaults() -> None:
    s = SchedulerConfig()
    assert s.name == "cosine"
    assert s.base_lr == 0.001
    assert s.min_lr == 1e-6
    assert s.warmup_epochs == 0
    assert s.warmup_start_lr == 1e-6
    assert s.params == {}


def test_scheduler_config_frozen() -> None:
    s = SchedulerConfig()
    with pytest.raises(Exception):
        s.name = "step"


def test_scheduler_config_custom() -> None:
    s = SchedulerConfig(name="step", base_lr=0.01, step_size=30, gamma=0.1)
    assert s.name == "step"
    assert s.base_lr == 0.01


def test_scheduler_config_serialization_roundtrip() -> None:
    s = SchedulerConfig(name="cosine", base_lr=0.01, warmup_epochs=5)
    data = s.model_dump()
    restored = SchedulerConfig(**data)
    assert restored.name == "cosine"
    assert restored.warmup_epochs == 5


# ── EarlyStoppingConfig ───────────────────────────────────────────────

def test_early_stopping_config_defaults() -> None:
    e = EarlyStoppingConfig()
    assert e.patience == 10
    assert e.mode == "min"
    assert e.min_delta == 0.001
    assert e.monitor == "validation_loss"
    assert e.restore_best_checkpoint is True


def test_early_stopping_config_frozen() -> None:
    e = EarlyStoppingConfig()
    with pytest.raises(Exception):
        e.patience = 5


def test_early_stopping_config_custom() -> None:
    e = EarlyStoppingConfig(patience=5, min_delta=0.01, mode="max")
    assert e.patience == 5
    assert e.min_delta == 0.01
    assert e.mode == "max"


def test_early_stopping_config_serialization_roundtrip() -> None:
    e = EarlyStoppingConfig(patience=20, mode="max")
    data = e.model_dump()
    restored = EarlyStoppingConfig(**data)
    assert restored.patience == 20
    assert restored.mode == "max"


# ── TrainingResult ────────────────────────────────────────────────────

def test_training_result_defaults() -> None:
    r = TrainingResult(experiment_id="exp_001", model_id="m_001")
    assert r.status == "completed"
    assert r.export_results == []
    assert r.total_epochs_completed == 0
    assert r.best_epoch is None
    assert r.best_metric is None
    assert r.best_metric_name == ""
    assert r.training_duration_seconds == 0.0
    assert r.final_metrics is None
    assert r.evaluation_result is None


def test_training_result_frozen() -> None:
    r = TrainingResult(experiment_id="exp_001", model_id="m_001")
    with pytest.raises(Exception):
        r.status = "failed"


def test_training_result_with_values() -> None:
    r = TrainingResult(
        experiment_id="exp_001", model_id="m_001",
        total_epochs_completed=50, best_epoch=45, best_metric=0.95,
        status="completed", training_duration_seconds=3600.0,
    )
    assert r.total_epochs_completed == 50
    assert r.best_metric == 0.95
    assert r.status == "completed"


def test_training_result_serialization_roundtrip() -> None:
    r = TrainingResult(experiment_id="exp_001", model_id="m_001", status="interrupted")
    data = r.model_dump()
    restored = TrainingResult(**data)
    assert restored.status == "interrupted"


# ── DatasetLoadResult ─────────────────────────────────────────────────

def test_dataset_load_result_defaults() -> None:
    d = DatasetLoadResult(
        dataset_name="coco", dataset_version="1.0",
        production_ready=True,
    )
    assert d.train_path == ""
    assert d.val_path == ""
    assert d.test_path == ""
    assert d.class_names == []
    assert d.num_classes == 0
    assert d.total_images == 0
    assert d.total_annotations == 0
    assert d.quality_report_version == ""


def test_dataset_load_result_frozen() -> None:
    d = DatasetLoadResult(
        dataset_name="coco", dataset_version="1.0",
        production_ready=True,
    )
    with pytest.raises(Exception):
        d.production_ready = False


def test_dataset_load_result_with_data() -> None:
    d = DatasetLoadResult(
        dataset_name="coco", dataset_version="2.0",
        production_ready=True, num_classes=80,
        total_images=100000, total_annotations=500000,
        class_names=["person", "car"],
        train_path="/data/train",
    )
    assert d.num_classes == 80
    assert d.total_images == 100000
    assert d.class_names == ["person", "car"]


def test_dataset_load_result_serialization_roundtrip() -> None:
    d = DatasetLoadResult(
        dataset_name="coco", dataset_version="1.0",
        production_ready=True, num_classes=80,
    )
    data = d.model_dump()
    restored = DatasetLoadResult(**data)
    assert restored.dataset_name == "coco"
    assert restored.production_ready is True
    assert restored.num_classes == 80


# ── AugmentationConfig ────────────────────────────────────────────────

def test_augmentation_config_defaults() -> None:
    a = AugmentationConfig()
    assert a.name == "default"
    assert a.flip_probability == 0.5
    assert a.rotate_degrees == (-10.0, 10.0)
    assert a.rotate_probability == 0.5
    assert a.scale_min == 0.5
    assert a.scale_max == 2.0
    assert a.blur_kernel_size == 5
    assert a.noise_intensity == 0.05
    assert a.mosaic_probability == 0.5
    assert a.mixup_probability == 0.5


def test_augmentation_config_frozen() -> None:
    a = AugmentationConfig()
    with pytest.raises(Exception):
        a.flip_probability = 0.0


def test_augmentation_config_ge_le_constraints() -> None:
    with pytest.raises(Exception):
        AugmentationConfig(flip_probability=-0.1)
    with pytest.raises(Exception):
        AugmentationConfig(flip_probability=1.5)


def test_augmentation_config_custom() -> None:
    a = AugmentationConfig(
        name="custom", flip_probability=0.3, blur_kernel_size=7,
    )
    assert a.name == "custom"
    assert a.flip_probability == 0.3
    assert a.blur_kernel_size == 7


def test_augmentation_config_serialization_roundtrip() -> None:
    a = AugmentationConfig(name="test", flip_probability=0.0)
    data = a.model_dump()
    restored = AugmentationConfig(**data)
    assert restored.name == "test"
    assert restored.flip_probability == 0.0


# ── HyperparameterProfile ─────────────────────────────────────────────

def test_hyperparameter_profile_defaults() -> None:
    p = HyperparameterProfile(name="test")
    assert p.learning_rate == 0.001
    assert p.batch_size == 16
    assert p.epochs == 100
    assert p.optimizer == "adam"
    assert p.scheduler == "cosine"
    assert p.image_size == (640, 640)
    assert p.weight_decay == 0.0001
    assert p.seed == 42
    assert p.early_stopping_patience is None
    assert p.mixed_precision is False
    assert p.description == ""


def test_hyperparameter_profile_frozen() -> None:
    p = HyperparameterProfile(name="test")
    with pytest.raises(Exception):
        p.learning_rate = 0.01


def test_hyperparameter_profile_gt_constraints() -> None:
    with pytest.raises(Exception):
        HyperparameterProfile(name="test", learning_rate=0)
    with pytest.raises(Exception):
        HyperparameterProfile(name="test", batch_size=0)
    with pytest.raises(Exception):
        HyperparameterProfile(name="test", epochs=0)


def test_hyperparameter_profile_custom() -> None:
    p = HyperparameterProfile(
        name="fast", learning_rate=0.01, batch_size=32,
        epochs=10, optimizer="sgd", description="quick",
    )
    assert p.name == "fast"
    assert p.batch_size == 32
    assert p.epochs == 10
    assert p.description == "quick"


def test_hyperparameter_profile_serialization_roundtrip() -> None:
    p = HyperparameterProfile(
        name="balanced", learning_rate=0.001, batch_size=16, epochs=100,
    )
    data = p.model_dump()
    restored = HyperparameterProfile(**data)
    assert restored.name == "balanced"
    assert restored.batch_size == 16
    assert restored.epochs == 100


# ── PreValidationResult ───────────────────────────────────────────────

def test_pre_validation_result_defaults() -> None:
    p = PreValidationResult(passed=True)
    assert p.passed is True
    assert p.errors == ()
    assert p.warnings == ()
    assert p.dataset_ready is False
    assert p.config_valid is False
    assert p.augmentation_valid is False
    assert p.num_errors == 0
    assert p.num_warnings == 0


def test_pre_validation_result_frozen() -> None:
    p = PreValidationResult(passed=True)
    with pytest.raises(Exception):
        p.passed = False


def test_pre_validation_result_with_errors() -> None:
    p = PreValidationResult(
        passed=False,
        errors=("batch_size must be >= 1", "epochs must be >= 1"),
        warnings=("batch_size > 512",),
        dataset_ready=True,
        config_valid=False,
        augmentation_valid=True,
        num_errors=2,
        num_warnings=1,
    )
    assert p.passed is False
    assert len(p.errors) == 2
    assert len(p.warnings) == 1
    assert p.dataset_ready is True
    assert p.config_valid is False
    assert p.num_errors == 2
    assert p.num_warnings == 1


def test_pre_validation_result_serialization_roundtrip() -> None:
    p = PreValidationResult(
        passed=False,
        errors=("error1",),
        num_errors=1,
    )
    data = p.model_dump()
    restored = PreValidationResult(**data)
    assert restored.passed is False
    assert restored.errors == ("error1",)
