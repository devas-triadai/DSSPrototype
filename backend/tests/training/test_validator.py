"""Tests for the TrainingValidator."""

from backend.training.models import TrainingConfigData
from backend.training.validator import TrainingValidator


def _make_validator() -> TrainingValidator:
    return TrainingValidator()


def test_validate_config_valid() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="yolo")
    result = validator.validate_config(config)
    assert result.config_valid is True
    assert result.errors == ()


def test_validate_config_empty_model_name() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="")
    result = validator.validate_config(config)
    assert result.passed is False
    assert "model_name is required" in result.errors


def test_validate_config_batch_size_zero() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", batch_size=0)
    result = validator.validate_config(config)
    assert "batch_size must be >= 1" in result.errors


def test_validate_config_batch_size_negative() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", batch_size=-1)
    result = validator.validate_config(config)
    assert "batch_size must be >= 1" in result.errors


def test_validate_config_batch_size_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", batch_size=1024)
    result = validator.validate_config(config)
    assert "batch_size > 512" in " ".join(result.warnings)


def test_validate_config_epochs_zero() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", epochs=0)
    result = validator.validate_config(config)
    assert "epochs must be >= 1" in result.errors


def test_validate_config_epochs_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", epochs=2000)
    result = validator.validate_config(config)
    assert "epochs > 1000" in " ".join(result.warnings)


def test_validate_config_learning_rate_zero() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", learning_rate=0.0)
    result = validator.validate_config(config)
    assert "learning_rate must be positive" in result.errors


def test_validate_config_learning_rate_negative() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", learning_rate=-0.5)
    result = validator.validate_config(config)
    assert "learning_rate must be positive" in result.errors


def test_validate_config_learning_rate_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", learning_rate=2.0)
    result = validator.validate_config(config)
    assert "learning_rate > 1.0" in " ".join(result.warnings)


def test_validate_config_invalid_optimizer_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", optimizer="unknown_opt")
    result = validator.validate_config(config)
    assert "optimizer" in " ".join(result.warnings).lower()


def test_validate_config_invalid_scheduler_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", scheduler="unknown_sch")
    result = validator.validate_config(config)
    assert "scheduler" in " ".join(result.warnings).lower()


def test_validate_config_invalid_device_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", device="tpu")
    result = validator.validate_config(config)
    assert "device" in " ".join(result.warnings).lower()


def test_validate_config_image_size_too_small() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", image_size=(16, 16))
    result = validator.validate_config(config)
    assert "image dimensions must be >= 32" in result.errors


def test_validate_config_image_size_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", image_size=(20000, 20000))
    result = validator.validate_config(config)
    assert "image dimensions > 10000" in " ".join(result.warnings)


def test_validate_config_weight_decay_negative() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", weight_decay=-0.1)
    result = validator.validate_config(config)
    assert "weight_decay must be non-negative" in result.errors


def test_validate_config_seed_negative() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", seed=-1)
    result = validator.validate_config(config)
    assert "seed must be non-negative" in result.errors


def test_validate_config_validation_interval_zero() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", validation_interval=0)
    result = validator.validate_config(config)
    assert "validation_interval must be >= 1" in result.errors


def test_validate_config_save_interval_zero() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", save_interval=0)
    result = validator.validate_config(config)
    assert "save_interval must be >= 1" in result.errors


def test_validate_config_early_stopping_patience_warning() -> None:
    validator = _make_validator()
    config = TrainingConfigData(model_name="test", early_stopping_patience=0)
    result = validator.validate_config(config)
    assert "early_stopping_patience" in " ".join(result.warnings)
