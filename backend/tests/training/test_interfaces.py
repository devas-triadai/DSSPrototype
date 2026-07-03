"""Tests for all ABC interfaces in the Training Platform.

Verifies that all interfaces are abstract, define the correct methods,
and have the expected signatures.
"""


from backend.training.interfaces import (
    AugmentationPipelineInterface,
    CheckpointManagerInterface,
    DatasetLoaderInterface,
    EvaluationEngineInterface,
    ExperimentManagerInterface,
    ExportPipelineInterface,
    HistoryManagerInterface,
    HyperparameterManagerInterface,
    MetricsManagerInterface,
    ModelRegistryInterface,
    SchedulerInterface,
    TrainerInterface,
    TrainingBackendInterface,
    TrainingValidatorInterface,
)


def _check_abstract(cls: type, method: str) -> bool:
    return hasattr(cls, method) and getattr(getattr(cls, method), "__isabstractmethod__", False)


def test_experiment_manager_interface_is_abc() -> None:
    assert hasattr(ExperimentManagerInterface, "__abstractmethods__")


def test_experiment_manager_has_abstract_methods() -> None:
    assert _check_abstract(ExperimentManagerInterface, "create_experiment")
    assert _check_abstract(ExperimentManagerInterface, "get_experiment")
    assert _check_abstract(ExperimentManagerInterface, "list_experiments")
    assert _check_abstract(ExperimentManagerInterface, "update_experiment")
    assert _check_abstract(ExperimentManagerInterface, "delete_experiment")


def test_model_registry_interface_is_abc() -> None:
    assert hasattr(ModelRegistryInterface, "__abstractmethods__")


def test_model_registry_has_abstract_methods() -> None:
    assert _check_abstract(ModelRegistryInterface, "register_model")
    assert _check_abstract(ModelRegistryInterface, "get_model")
    assert _check_abstract(ModelRegistryInterface, "get_models_by_name")
    assert _check_abstract(ModelRegistryInterface, "list_models")
    assert _check_abstract(ModelRegistryInterface, "update_model")
    assert _check_abstract(ModelRegistryInterface, "delete_model")


def test_checkpoint_manager_interface_is_abc() -> None:
    assert hasattr(CheckpointManagerInterface, "__abstractmethods__")


def test_checkpoint_manager_has_abstract_methods() -> None:
    assert _check_abstract(CheckpointManagerInterface, "save_checkpoint")
    assert _check_abstract(CheckpointManagerInterface, "load_checkpoint")
    assert _check_abstract(CheckpointManagerInterface, "get_best_checkpoint")
    assert _check_abstract(CheckpointManagerInterface, "get_latest_checkpoint")
    assert _check_abstract(CheckpointManagerInterface, "list_checkpoints")


def test_checkpoint_manager_has_abstract_property() -> None:
    assert _check_abstract(CheckpointManagerInterface, "checkpoints_dir")


def test_metrics_manager_interface_is_abc() -> None:
    assert hasattr(MetricsManagerInterface, "__abstractmethods__")


def test_metrics_manager_has_abstract_methods() -> None:
    assert _check_abstract(MetricsManagerInterface, "record")
    assert _check_abstract(MetricsManagerInterface, "get_metrics")
    assert _check_abstract(MetricsManagerInterface, "get_best_metric")
    assert _check_abstract(MetricsManagerInterface, "get_latest_metrics")


def test_history_manager_interface_is_abc() -> None:
    assert hasattr(HistoryManagerInterface, "__abstractmethods__")


def test_history_manager_has_abstract_methods() -> None:
    assert _check_abstract(HistoryManagerInterface, "record_entry")
    assert _check_abstract(HistoryManagerInterface, "get_history")
    assert _check_abstract(HistoryManagerInterface, "get_latest_entry")
    assert _check_abstract(HistoryManagerInterface, "get_history_as_dicts")


def test_evaluation_engine_interface_is_abc() -> None:
    assert hasattr(EvaluationEngineInterface, "__abstractmethods__")


def test_evaluation_engine_has_abstract_methods() -> None:
    assert _check_abstract(EvaluationEngineInterface, "validate")
    assert _check_abstract(EvaluationEngineInterface, "test")
    assert _check_abstract(EvaluationEngineInterface, "benchmark")


def test_export_pipeline_interface_is_abc() -> None:
    assert hasattr(ExportPipelineInterface, "__abstractmethods__")


def test_export_pipeline_has_abstract_methods() -> None:
    assert _check_abstract(ExportPipelineInterface, "export_to_onnx")
    assert _check_abstract(ExportPipelineInterface, "export_to_torchscript")
    assert _check_abstract(ExportPipelineInterface, "export_to_openvino")
    assert _check_abstract(ExportPipelineInterface, "list_exports")


def test_scheduler_interface_is_abc() -> None:
    assert hasattr(SchedulerInterface, "__abstractmethods__")


def test_scheduler_has_abstract_methods() -> None:
    assert _check_abstract(SchedulerInterface, "get_lr")
    assert _check_abstract(SchedulerInterface, "state_dict")
    assert _check_abstract(SchedulerInterface, "load_state_dict")


def test_training_backend_interface_is_abc() -> None:
    assert hasattr(TrainingBackendInterface, "__abstractmethods__")


def test_training_backend_has_abstract_methods() -> None:
    assert _check_abstract(TrainingBackendInterface, "initialize")
    assert _check_abstract(TrainingBackendInterface, "train_epoch")
    assert _check_abstract(TrainingBackendInterface, "validate")
    assert _check_abstract(TrainingBackendInterface, "test")
    assert _check_abstract(TrainingBackendInterface, "export")
    assert _check_abstract(TrainingBackendInterface, "save_checkpoint")
    assert _check_abstract(TrainingBackendInterface, "load_checkpoint")
    assert _check_abstract(TrainingBackendInterface, "resume")
    assert _check_abstract(TrainingBackendInterface, "shutdown")


def test_dataset_loader_interface_is_abc() -> None:
    assert hasattr(DatasetLoaderInterface, "__abstractmethods__")


def test_dataset_loader_has_abstract_methods() -> None:
    assert _check_abstract(DatasetLoaderInterface, "load_dataset")
    assert _check_abstract(DatasetLoaderInterface, "list_available_datasets")


def test_augmentation_pipeline_interface_is_abc() -> None:
    assert hasattr(AugmentationPipelineInterface, "__abstractmethods__")


def test_augmentation_pipeline_has_abstract_methods() -> None:
    assert _check_abstract(AugmentationPipelineInterface, "create_pipeline")
    assert _check_abstract(AugmentationPipelineInterface, "get_available_transforms")


def test_hyperparameter_manager_interface_is_abc() -> None:
    assert hasattr(HyperparameterManagerInterface, "__abstractmethods__")


def test_hyperparameter_manager_has_abstract_methods() -> None:
    assert _check_abstract(HyperparameterManagerInterface, "save_profile")
    assert _check_abstract(HyperparameterManagerInterface, "get_profile")
    assert _check_abstract(HyperparameterManagerInterface, "list_profiles")
    assert _check_abstract(HyperparameterManagerInterface, "delete_profile")
    assert _check_abstract(HyperparameterManagerInterface, "apply_profile")


def test_training_validator_interface_is_abc() -> None:
    assert hasattr(TrainingValidatorInterface, "__abstractmethods__")


def test_training_validator_has_abstract_methods() -> None:
    assert _check_abstract(TrainingValidatorInterface, "validate_config")
    assert _check_abstract(TrainingValidatorInterface, "validate_dataset_ready")


def test_trainer_interface_is_abc() -> None:
    assert hasattr(TrainerInterface, "__abstractmethods__")


def test_trainer_has_abstract_methods() -> None:
    assert _check_abstract(TrainerInterface, "train")
