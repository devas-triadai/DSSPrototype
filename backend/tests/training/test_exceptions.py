"""Tests for the TrainingError exception hierarchy."""

import pytest

from backend.training.exceptions import (
    AugmentationError,
    CheckpointNotFoundError,
    ConfigurationError,
    DatasetLoadError,
    DatasetNotReadyError,
    EvaluationError,
    ExperimentNotFoundError,
    ExportError,
    ModelNotFoundError,
    PreValidationError,
    TrainingError,
    TrainingInterruptedError,
)


def test_training_error_is_base() -> None:
    assert issubclass(ExperimentNotFoundError, TrainingError)
    assert issubclass(ModelNotFoundError, TrainingError)
    assert issubclass(CheckpointNotFoundError, TrainingError)
    assert issubclass(ConfigurationError, TrainingError)
    assert issubclass(ExportError, TrainingError)
    assert issubclass(EvaluationError, TrainingError)
    assert issubclass(TrainingInterruptedError, TrainingError)
    assert issubclass(DatasetNotReadyError, TrainingError)
    assert issubclass(DatasetLoadError, TrainingError)
    assert issubclass(AugmentationError, TrainingError)
    assert issubclass(PreValidationError, TrainingError)


def test_training_error_is_exception() -> None:
    assert issubclass(TrainingError, Exception)


def test_experiment_not_found_raise_and_catch() -> None:
    with pytest.raises(ExperimentNotFoundError):
        raise ExperimentNotFoundError("Experiment not found: exp_001")


def test_model_not_found_raise_and_catch() -> None:
    with pytest.raises(ModelNotFoundError):
        raise ModelNotFoundError("Model not found: m_001")


def test_checkpoint_not_found_raise_and_catch() -> None:
    with pytest.raises(CheckpointNotFoundError):
        raise CheckpointNotFoundError("Checkpoint not found")


def test_configuration_error_raise_and_catch() -> None:
    with pytest.raises(ConfigurationError):
        raise ConfigurationError("Invalid configuration")


def test_export_error_raise_and_catch() -> None:
    with pytest.raises(ExportError):
        raise ExportError("Export failed")


def test_evaluation_error_raise_and_catch() -> None:
    with pytest.raises(EvaluationError):
        raise EvaluationError("Evaluation failed")


def test_training_interrupted_raise_and_catch() -> None:
    with pytest.raises(TrainingInterruptedError):
        raise TrainingInterruptedError("Training interrupted by callback")


def test_dataset_not_ready_raise_and_catch() -> None:
    with pytest.raises(DatasetNotReadyError):
        raise DatasetNotReadyError("Dataset is not production ready")


def test_dataset_load_error_raise_and_catch() -> None:
    with pytest.raises(DatasetLoadError):
        raise DatasetLoadError("Failed to load dataset")


def test_augmentation_error_raise_and_catch() -> None:
    with pytest.raises(AugmentationError):
        raise AugmentationError("Augmentation failed")


def test_pre_validation_error_raise_and_catch() -> None:
    with pytest.raises(PreValidationError):
        raise PreValidationError("Pre-validation failed")


def test_experiment_not_found_is_training_error() -> None:
    try:
        raise ExperimentNotFoundError("test")
    except TrainingError:
        assert True
    except Exception:
        assert False


def test_error_with_message() -> None:
    msg = "Custom error message"
    try:
        raise ConfigurationError(msg)
    except TrainingError as e:
        assert str(e) == msg


def test_all_exception_types_are_distinct() -> None:
    types = [
        ExperimentNotFoundError,
        ModelNotFoundError,
        CheckpointNotFoundError,
        ConfigurationError,
        ExportError,
        EvaluationError,
        TrainingInterruptedError,
        DatasetNotReadyError,
        DatasetLoadError,
        AugmentationError,
        PreValidationError,
    ]
    assert len(set(types)) == len(types)
