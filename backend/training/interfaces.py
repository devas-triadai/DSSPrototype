"""Abstract interfaces for every component in the Training Platform.

All concrete implementations depend on these contracts, never on each other.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

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

# Forward reference for type hints in TrainingBackendInterface.
EpochCallback = Callable[[int, dict[str, float]], None]


class ExperimentManagerInterface(ABC):
    """Contract for experiment lifecycle management."""

    @abstractmethod
    def create_experiment(self, config: TrainingConfigData) -> ExperimentData:
        """Create a new experiment and return its metadata."""

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> ExperimentData | None:
        """Retrieve an experiment by ID."""

    @abstractmethod
    def list_experiments(self) -> list[ExperimentData]:
        """Return all experiments, newest first."""

    @abstractmethod
    def update_experiment(self, experiment: ExperimentData) -> ExperimentData:
        """Update experiment metadata (status, best_metric, etc.)."""

    @abstractmethod
    def delete_experiment(self, experiment_id: str) -> bool:
        """Remove an experiment."""


class ModelRegistryInterface(ABC):
    """Contract for the model registry."""

    @abstractmethod
    def register_model(self, entry: ModelEntry) -> ModelEntry:
        """Register a new model."""

    @abstractmethod
    def get_model(self, model_id: str) -> ModelEntry | None:
        """Retrieve a model by ID."""

    @abstractmethod
    def get_models_by_name(self, name: str) -> list[ModelEntry]:
        """Retrieve all versions of a model by name."""

    @abstractmethod
    def list_models(self) -> list[ModelEntry]:
        """Return all registered models."""

    @abstractmethod
    def update_model(self, entry: ModelEntry) -> ModelEntry:
        """Update model metadata."""

    @abstractmethod
    def delete_model(self, model_id: str) -> bool:
        """Remove a model from the registry."""


class CheckpointManagerInterface(ABC):
    """Contract for checkpoint persistence."""

    @property
    @abstractmethod
    def checkpoints_dir(self) -> Path:
        """Return the base directory where checkpoints are stored."""

    @abstractmethod
    def save_checkpoint(
        self,
        experiment_id: str,
        epoch: int,
        metric_value: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CheckpointData:
        """Save a checkpoint for the given epoch.

        Returns metadata about the saved checkpoint.
        """

    @abstractmethod
    def load_checkpoint(self, experiment_id: str, epoch: int) -> CheckpointData | None:
        """Load checkpoint metadata for a specific epoch."""

    @abstractmethod
    def get_best_checkpoint(self, experiment_id: str) -> CheckpointData | None:
        """Return the best checkpoint by metric value."""

    @abstractmethod
    def get_latest_checkpoint(self, experiment_id: str) -> CheckpointData | None:
        """Return the most recent checkpoint."""

    @abstractmethod
    def list_checkpoints(self, experiment_id: str) -> list[CheckpointData]:
        """Return all checkpoints for an experiment, ordered by epoch."""


class MetricsManagerInterface(ABC):
    """Contract for training metrics collection and storage."""

    @abstractmethod
    def record(self, metric: MetricData) -> MetricData:
        """Record a metric snapshot."""

    @abstractmethod
    def get_metrics(
        self, experiment_id: str, epoch: int | None = None,
    ) -> list[MetricData]:
        """Retrieve metrics, optionally filtered by epoch."""

    @abstractmethod
    def get_best_metric(
        self, experiment_id: str, metric_name: str = "validation_loss", mode: str = "min",
    ) -> MetricData | None:
        """Return the metric snapshot with the best value."""

    @abstractmethod
    def get_latest_metrics(self, experiment_id: str) -> MetricData | None:
        """Return the most recent metric snapshot."""


class HistoryManagerInterface(ABC):
    """Contract for persisting training history."""

    @abstractmethod
    def record_entry(self, entry: HistoryEntry) -> HistoryEntry:
        """Record a history entry."""

    @abstractmethod
    def get_history(self, experiment_id: str) -> list[HistoryEntry]:
        """Return all history entries for an experiment, ordered by epoch."""

    @abstractmethod
    def get_latest_entry(self, experiment_id: str) -> HistoryEntry | None:
        """Return the most recent history entry."""

    @abstractmethod
    def get_history_as_dicts(self, experiment_id: str) -> list[dict[str, object]]:
        """Return history as a list of dicts, suitable for plotting."""


class EvaluationEngineInterface(ABC):
    """Contract for model evaluation."""

    @abstractmethod
    def validate(
        self, experiment_id: str, checkpoint_path: str, dataset_version: str = "",
    ) -> EvaluationResult:
        """Run validation on a model checkpoint."""

    @abstractmethod
    def test(
        self, experiment_id: str, checkpoint_path: str, dataset_version: str = "",
    ) -> EvaluationResult:
        """Run test evaluation on a model checkpoint."""

    @abstractmethod
    def benchmark(
        self, experiment_id: str, checkpoint_path: str,
    ) -> EvaluationResult:
        """Run performance benchmarking on a model checkpoint."""


class ExportPipelineInterface(ABC):
    """Contract for model export."""

    @abstractmethod
    def export_to_onnx(self, experiment_id: str, model_id: str, output_dir: Path) -> ExportData:
        """Export a model to ONNX format."""

    @abstractmethod
    def export_to_torchscript(
        self, experiment_id: str, model_id: str, output_dir: Path,
    ) -> ExportData:
        """Export a model to TorchScript format."""

    @abstractmethod
    def export_to_openvino(self, experiment_id: str, model_id: str, output_dir: Path) -> ExportData:
        """Export a model to OpenVINO format."""

    @abstractmethod
    def list_exports(self, model_id: str) -> list[ExportData]:
        """Return all exports for a model."""


class SchedulerInterface(ABC):
    """Contract for learning rate schedulers."""

    @abstractmethod
    def get_lr(self, epoch: int, base_lr: float) -> float:
        """Compute the learning rate for a given epoch."""

    @abstractmethod
    def state_dict(self) -> dict[str, object]:
        """Return scheduler state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore scheduler state from a checkpoint."""


class TrainingBackendInterface(ABC):
    """Contract for model-specific training backends.

    Every backend (YOLO, Detectron2, MMDetection, etc.) implements
    this interface. The Trainer depends ONLY on this interface —
    never on concrete backend classes.
    """

    @abstractmethod
    def initialize(
        self,
        config: TrainingConfigData,
        experiment_id: str,
        dataset_path: str,
    ) -> None:
        """Prepare the backend for training.

        Sets up model architecture, optimizer, dataset paths, and
        any framework-specific state. Called once before the training loop.
        """

    @abstractmethod
    def train_epoch(self, epoch: int, learning_rate: float) -> dict[str, float]:
        """Run one training epoch.

        Parameters
        ----------
        epoch:
            Current epoch number (1-indexed).
        learning_rate:
            Learning rate to use for this epoch (computed by Scheduler).

        Returns
        -------
        dict[str, float]
            Metrics from this epoch: training_loss, learning_rate,
            epoch_time_ms, etc.
        """

    @abstractmethod
    def validate(
        self,
        checkpoint_path: str,
        dataset_path: str,
        experiment_id: str = "",
    ) -> EvaluationResult:
        """Run validation evaluation on a checkpoint."""

    @abstractmethod
    def test(
        self,
        checkpoint_path: str,
        dataset_path: str,
        experiment_id: str = "",
    ) -> EvaluationResult:
        """Run test evaluation on a checkpoint."""

    @abstractmethod
    def export(
        self,
        checkpoint_path: str,
        format_name: str,
        output_dir: Path,
        model_id: str = "",
        experiment_id: str = "",
    ) -> str:
        """Export a trained model to the specified format.

        Returns the path to the exported model file.
        """

    @abstractmethod
    def save_checkpoint(self, output_path: str, epoch: int) -> dict[str, object]:
        """Save framework-specific model weights.

        Returns metadata dict (state dict locations, file sizes, etc.).
        """

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load framework-specific model weights into the backend."""

    @abstractmethod
    def resume(self, checkpoint_path: str) -> int:
        """Resume training from a checkpoint.

        Returns the epoch to resume from (e.g., last completed epoch + 1).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release all framework resources (GPU memory, processes, etc.)."""


class DatasetLoaderInterface(ABC):
    """Contract for loading datasets from the Dataset Quality pipeline.

    Only production-approved datasets may pass through.
    """

    @abstractmethod
    def load_dataset(
        self,
        dataset_name: str,
        dataset_version: str = "",
    ) -> DatasetLoadResult:
        """Load a dataset that has passed Dataset Quality checks.

        Raises DatasetNotReadyError if the dataset is not production_ready.
        """

    @abstractmethod
    def list_available_datasets(self) -> list[DatasetLoadResult]:
        """List all datasets available and approved for training."""


class AugmentationPipelineInterface(ABC):
    """Contract for configurable augmentation pipelines.

    Supports multiple augmentation transforms applied in sequence,
    each with configurable probability and intensity.
    """

    @abstractmethod
    def create_pipeline(self, config: AugmentationConfig) -> object:
        """Create an augmentation pipeline from a configuration.

        Returns a framework-specific pipeline object.
        """

    @abstractmethod
    def get_available_transforms(self) -> list[str]:
        """Return the list of available augmentation transforms."""


class HyperparameterManagerInterface(ABC):
    """Contract for managing named hyperparameter profiles."""

    @abstractmethod
    def save_profile(self, profile: HyperparameterProfile) -> HyperparameterProfile:
        """Save a named hyperparameter profile."""

    @abstractmethod
    def get_profile(self, name: str) -> HyperparameterProfile | None:
        """Retrieve a profile by name."""

    @abstractmethod
    def list_profiles(self) -> list[HyperparameterProfile]:
        """Return all saved profiles."""

    @abstractmethod
    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name."""

    @abstractmethod
    def apply_profile(
        self, profile: HyperparameterProfile, overrides: dict[str, object] | None = None,
    ) -> TrainingConfigData:
        """Apply a profile to produce a TrainingConfigData with optional overrides."""


class TrainingValidatorInterface(ABC):
    """Contract for validating training configuration before execution."""

    @abstractmethod
    def validate_config(self, config: TrainingConfigData) -> PreValidationResult:
        """Validate that a training configuration is valid before execution."""

    @abstractmethod
    def validate_dataset_ready(self, dataset_name: str, dataset_version: str) -> bool:
        """Check whether a dataset is ready for training."""


class TrainerInterface(ABC):
    """Contract for the training coordinator.

    The Trainer depends ONLY on interfaces.
    """

    @abstractmethod
    def train(
        self,
        config: TrainingConfigData,
        dataset_metadata: object | None = None,
    ) -> TrainingResult:
        """Execute a full training run.

        Parameters
        ----------
        config:
            Complete training configuration.
        dataset_metadata:
            Metadata from Dataset Management Platform (optional).

        Returns
        -------
        TrainingResult
            Complete result including metrics, exports, evaluation.
        """
