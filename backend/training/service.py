"""TrainingService — public entry point for the Training Platform.

Coordinates experiment management, model registry, training
execution, dataset loading, augmentation, hyperparameter profiles,
validation, and pipeline orchestration.
"""

import logging
from pathlib import Path

from backend.training.augmentation import AugmentationPipeline
from backend.training.callbacks import Callback
from backend.training.checkpoint import CheckpointManager
from backend.training.dataset_loader import DatasetLoader
from backend.training.evaluator import EvaluationEngine
from backend.training.exceptions import (
    ExperimentNotFoundError,
)
from backend.training.experiment import ExperimentManager
from backend.training.exporter import ExportPipeline
from backend.training.history import HistoryManager
from backend.training.hyperparameter_manager import HyperparameterManager
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
    TrainingBackendInterface,
    TrainingValidatorInterface,
)
from backend.training.metrics import MetricsManager
from backend.training.models import (
    AugmentationConfig,
    CheckpointData,
    DatasetLoadResult,
    EvaluationResult,
    ExperimentData,
    ExportData,
    HistoryEntry,
    HyperparameterProfile,
    MetricData,
    ModelEntry,
    PreValidationResult,
    TrainingConfigData,
    TrainingResult,
)
from backend.training.registry import ModelRegistry
from backend.training.scheduler import Scheduler
from backend.training.trainer import Trainer
from backend.training.training_pipeline import TrainingPipeline
from backend.training.validator import TrainingValidator

logger = logging.getLogger("dss.training.service")


class TrainingService:
    """Public entry point for the Training Platform.

    Provides high-level access to experiment management, model registry,
    checkpoint management, metrics, history, evaluation, and export.

    Integrates with DatasetManagementService via dataset_metadata.
    """

    def __init__(
        self,
        experiment_manager: ExperimentManagerInterface | None = None,
        model_registry: ModelRegistryInterface | None = None,
        checkpoint_manager: CheckpointManagerInterface | None = None,
        metrics_manager: MetricsManagerInterface | None = None,
        history_manager: HistoryManagerInterface | None = None,
        evaluation_engine: EvaluationEngineInterface | None = None,
        export_pipeline: ExportPipelineInterface | None = None,
        scheduler: SchedulerInterface | None = None,
        training_backend: TrainingBackendInterface | None = None,
        dataset_loader: DatasetLoaderInterface | None = None,
        augmentation_pipeline: AugmentationPipelineInterface | None = None,
        hyperparameter_manager: HyperparameterManagerInterface | None = None,
        validator: TrainingValidatorInterface | None = None,
    ) -> None:
        self._experiment_manager = experiment_manager or ExperimentManager()
        self._model_registry = model_registry or ModelRegistry()
        self._checkpoint_manager = checkpoint_manager or CheckpointManager()
        self._metrics_manager = metrics_manager or MetricsManager()
        self._history_manager = history_manager or HistoryManager()
        self._evaluation_engine = evaluation_engine or EvaluationEngine()
        self._export_pipeline = export_pipeline or ExportPipeline()
        self._scheduler = scheduler or Scheduler()
        self._training_backend = training_backend
        self._dataset_loader = dataset_loader or DatasetLoader()
        self._augmentation_pipeline = augmentation_pipeline or AugmentationPipeline()
        self._hyperparameter_manager = hyperparameter_manager or HyperparameterManager()
        self._validator = validator or TrainingValidator()

    # ------------------------------------------------------------------
    # Training execution
    # ------------------------------------------------------------------

    def train(
        self,
        config: TrainingConfigData,
        callbacks: list[Callback] | None = None,
        dataset_metadata: object | None = None,
    ) -> TrainingResult:
        """Execute a training run with the given configuration.

        Creates a Trainer with all injected dependencies and runs it.
        """
        logger.info("Training service: starting experiment '%s'", config.experiment_name)

        if self._training_backend is None:
            raise RuntimeError(
                "No training backend configured. "
                "Set a backend via set_training_backend() or pass one to the constructor.",
            )

        trainer = Trainer(
            experiment_manager=self._experiment_manager,
            model_registry=self._model_registry,
            checkpoint_manager=self._checkpoint_manager,
            metrics_manager=self._metrics_manager,
            history_manager=self._history_manager,
            evaluation_engine=self._evaluation_engine,
            export_pipeline=self._export_pipeline,
            training_backend=self._training_backend,
            scheduler=self._scheduler,
            callbacks=callbacks,
        )

        return trainer.train(config, dataset_metadata)

    def set_training_backend(self, backend: TrainingBackendInterface) -> None:
        """Set or replace the training backend."""
        self._training_backend = backend
        logger.info("Training backend set: %s", type(backend).__name__)

    # ------------------------------------------------------------------
    # Pipeline orchestration
    # ------------------------------------------------------------------

    def run_pipeline(
        self,
        config: TrainingConfigData,
        augmentation_config: AugmentationConfig | None = None,
        dataset_metadata: object | None = None,
    ) -> TrainingResult:
        """Execute the full training pipeline.

        Validates dataset → loads → augments → validates config →
        trains → evaluates → checkpoints → registers → exports.
        """
        if self._training_backend is None:
            raise RuntimeError("No training backend configured")
        pipeline = TrainingPipeline(
            dataset_loader=self._dataset_loader,
            augmentation_pipeline=self._augmentation_pipeline,
            hyperparameter_manager=self._hyperparameter_manager,
            validator=self._validator,
            experiment_manager=self._experiment_manager,
            model_registry=self._model_registry,
            checkpoint_manager=self._checkpoint_manager,
            metrics_manager=self._metrics_manager,
            history_manager=self._history_manager,
            evaluation_engine=self._evaluation_engine,
            export_pipeline=self._export_pipeline,
            training_backend=self._training_backend,
            scheduler=self._scheduler,
        )
        return pipeline.run(config, augmentation_config, dataset_metadata)

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_training_dataset(
        self, dataset_name: str, dataset_version: str = "",
    ) -> DatasetLoadResult:
        """Load a production-approved dataset for training."""
        return self._dataset_loader.load_dataset(dataset_name, dataset_version)

    def list_training_datasets(self) -> list[DatasetLoadResult]:
        """List all datasets approved for training."""
        return self._dataset_loader.list_available_datasets()

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def create_augmentation_pipeline(
        self, config: AugmentationConfig,
    ) -> object:
        """Create an augmentation pipeline configuration."""
        return self._augmentation_pipeline.create_pipeline(config)

    def list_augmentation_transforms(self) -> list[str]:
        """List available augmentation transforms."""
        return self._augmentation_pipeline.get_available_transforms()

    # ------------------------------------------------------------------
    # Hyperparameter profiles
    # ------------------------------------------------------------------

    def save_hyperparameter_profile(self, profile: HyperparameterProfile) -> HyperparameterProfile:
        return self._hyperparameter_manager.save_profile(profile)

    def get_hyperparameter_profile(self, name: str) -> HyperparameterProfile | None:
        return self._hyperparameter_manager.get_profile(name)

    def list_hyperparameter_profiles(self) -> list[HyperparameterProfile]:
        return self._hyperparameter_manager.list_profiles()

    def delete_hyperparameter_profile(self, name: str) -> bool:
        return self._hyperparameter_manager.delete_profile(name)

    def apply_hyperparameter_profile(
        self, profile: HyperparameterProfile, overrides: dict[str, object] | None = None,
    ) -> TrainingConfigData:
        return self._hyperparameter_manager.apply_profile(profile, overrides)

    # ------------------------------------------------------------------
    # Pre-training validation
    # ------------------------------------------------------------------

    def validate_training_config(self, config: TrainingConfigData) -> PreValidationResult:
        return self._validator.validate_config(config)

    def validate_dataset_ready(self, dataset_name: str, dataset_version: str) -> bool:
        return self._validator.validate_dataset_ready(dataset_name, dataset_version)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def create_experiment(self, config: TrainingConfigData) -> ExperimentData:
        return self._experiment_manager.create_experiment(config)

    def get_experiment(self, experiment_id: str) -> ExperimentData | None:
        return self._experiment_manager.get_experiment(experiment_id)

    def list_experiments(self) -> list[ExperimentData]:
        return self._experiment_manager.list_experiments()

    def update_experiment_status(
        self, experiment_id: str, status: str,
    ) -> ExperimentData:
        experiment = self._experiment_manager.get_experiment(experiment_id)
        if experiment is None:
            raise ExperimentNotFoundError(f"Experiment not found: {experiment_id}")
        updated = ExperimentData(
            experiment_id=experiment.experiment_id,
            experiment_name=experiment.experiment_name,
            dataset_version=experiment.dataset_version,
            config=experiment.config,
            training_start=experiment.training_start,
            training_end=experiment.training_end,
            duration_seconds=experiment.duration_seconds,
            best_epoch=experiment.best_epoch,
            best_metric=experiment.best_metric,
            best_metric_name=experiment.best_metric_name,
            status=status,
            notes=experiment.notes,
        )
        return self._experiment_manager.update_experiment(updated)

    # ------------------------------------------------------------------
    # Model registry
    # ------------------------------------------------------------------

    def register_model(self, entry: ModelEntry) -> ModelEntry:
        return self._model_registry.register_model(entry)

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self._model_registry.get_model(model_id)

    def list_models(self) -> list[ModelEntry]:
        return self._model_registry.list_models()

    def update_model(self, entry: ModelEntry) -> ModelEntry:
        return self._model_registry.update_model(entry)

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def list_checkpoints(self, experiment_id: str) -> list[CheckpointData]:
        return self._checkpoint_manager.list_checkpoints(experiment_id)

    def get_best_checkpoint(self, experiment_id: str) -> CheckpointData | None:
        return self._checkpoint_manager.get_best_checkpoint(experiment_id)

    def get_latest_checkpoint(self, experiment_id: str) -> CheckpointData | None:
        return self._checkpoint_manager.get_latest_checkpoint(experiment_id)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(
        self, experiment_id: str, epoch: int | None = None,
    ) -> list[MetricData]:
        return self._metrics_manager.get_metrics(experiment_id, epoch)

    def get_best_metric(
        self, experiment_id: str, metric_name: str = "validation_loss",
        mode: str = "min",
    ) -> MetricData | None:
        return self._metrics_manager.get_best_metric(
            experiment_id, metric_name, mode,
        )

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def get_history(self, experiment_id: str) -> list[HistoryEntry]:
        return self._history_manager.get_history(experiment_id)

    def get_history_as_dicts(self, experiment_id: str) -> list[dict[str, object]]:
        return self._history_manager.get_history_as_dicts(experiment_id)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, experiment_id: str, checkpoint_path: str,
        dataset_version: str = "",
    ) -> EvaluationResult:
        return self._evaluation_engine.validate(
            experiment_id, checkpoint_path, dataset_version,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(
        self, experiment_id: str, model_id: str, output_dir: Path,
        formats: list[str] | None = None,
    ) -> list[ExportData]:
        exports: list[ExportData] = []
        fmt_list = formats or ["onnx"]
        for fmt in fmt_list:
            if fmt == "onnx":
                exports.append(
                    self._export_pipeline.export_to_onnx(
                        experiment_id, model_id, output_dir,
                    ),
                )
            elif fmt == "torchscript":
                exports.append(
                    self._export_pipeline.export_to_torchscript(
                        experiment_id, model_id, output_dir,
                    ),
                )
            elif fmt == "openvino":
                exports.append(
                    self._export_pipeline.export_to_openvino(
                        experiment_id, model_id, output_dir,
                    ),
                )
        return exports
