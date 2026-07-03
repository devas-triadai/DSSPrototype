"""Tests for TrainingConfig."""

from pathlib import Path

from backend.training.config import TrainingConfig, training_config


def test_config_defaults() -> None:
    cfg = TrainingConfig()
    assert cfg.default_batch_size == 16
    assert cfg.default_epochs == 100
    assert cfg.default_learning_rate == 0.001
    assert cfg.default_optimizer == "adam"
    assert cfg.default_image_size == (640, 640)
    assert cfg.default_seed == 42
    assert cfg.default_workers == 4
    assert cfg.default_mixed_precision is False
    assert cfg.default_device == "cpu"


def test_env_prefix() -> None:
    assert TrainingConfig.model_config.get("env_prefix") == "TR_"


def test_singleton_exists() -> None:
    assert training_config is not None
    assert isinstance(training_config, TrainingConfig)


def test_singleton_is_same_instance() -> None:
    from backend.training.config import training_config as tc2
    assert training_config is tc2


def test_base_dir_is_path() -> None:
    assert isinstance(training_config.base_dir, Path)


def test_base_dir_exists() -> None:
    assert training_config.base_dir.exists()


def test_training_root_is_under_base_dir() -> None:
    assert str(training_config.training_root).startswith(str(training_config.base_dir))


def test_experiments_dir_path() -> None:
    assert isinstance(training_config.experiments_dir, Path)


def test_models_dir_path() -> None:
    assert isinstance(training_config.models_dir, Path)


def test_checkpoints_dir_path() -> None:
    assert isinstance(training_config.checkpoints_dir, Path)


def test_metrics_dir_path() -> None:
    assert isinstance(training_config.metrics_dir, Path)


def test_logs_dir_path() -> None:
    assert isinstance(training_config.logs_dir, Path)


def test_exports_dir_path() -> None:
    assert isinstance(training_config.exports_dir, Path)


def test_reports_dir_path() -> None:
    assert isinstance(training_config.reports_dir, Path)


def test_history_dir_path() -> None:
    assert isinstance(training_config.history_dir, Path)


def test_configs_dir_path() -> None:
    assert isinstance(training_config.configs_dir, Path)


def test_save_interval_default() -> None:
    assert training_config.save_interval == 5


def test_validation_interval_default() -> None:
    assert training_config.validation_interval == 1


def test_keep_last_n_checkpoints_default() -> None:
    assert training_config.keep_last_n_checkpoints == 3


def test_early_stopping_patience_default() -> None:
    assert training_config.early_stopping_patience == 10


def test_early_stopping_delta_default() -> None:
    assert training_config.early_stopping_delta == 0.001


def test_export_onnx_opset_default() -> None:
    assert training_config.export_onnx_opset == 17


def test_export_torchscript_method_default() -> None:
    assert training_config.export_torchscript_method == "trace"


def test_config_overridable_via_env() -> None:
    cfg = TrainingConfig(default_batch_size=32)
    assert cfg.default_batch_size == 32
