"""AI Training Platform — infrastructure for training all future CV models.

Provides experiment tracking, model registry, checkpoint management,
metrics, evaluation, export, callbacks, early stopping, scheduler,
dataset loading, augmentation, hyperparameter management, validation,
and pipeline orchestration — all model-agnostic and framework-agnostic.
"""

from backend.training.augmentation import AugmentationPipeline, TransformDescriptor
from backend.training.backends import TrainingBackendRegistry, YOLOTrainingBackend
from backend.training.callbacks import Callback, CallbackRunner
from backend.training.checkpoint import CheckpointManager
from backend.training.config import TrainingConfig, training_config
from backend.training.dataset_loader import DatasetLoader
from backend.training.early_stopping import EarlyStopping
from backend.training.evaluator import EvaluationEngine
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
    TrainerInterface,
    TrainingBackendInterface,
    TrainingValidatorInterface,
)
from backend.training.metrics import MetricsManager
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
from backend.training.registry import ModelRegistry
from backend.training.scheduler import Scheduler
from backend.training.service import TrainingService
from backend.training.trainer import Trainer
from backend.training.training_pipeline import TrainingPipeline
from backend.training.validator import TrainingValidator

__all__ = [
    "TrainingConfig",
    "training_config",
    "TrainingConfigData",
    "ExperimentData",
    "ModelEntry",
    "CheckpointData",
    "MetricData",
    "EvaluationResult",
    "ExportData",
    "HistoryEntry",
    "TrainingResult",
    "SchedulerConfig",
    "EarlyStoppingConfig",
    "DatasetLoadResult",
    "AugmentationConfig",
    "HyperparameterProfile",
    "PreValidationResult",
    "TrainerInterface",
    "TrainingBackendInterface",
    "ExperimentManagerInterface",
    "ModelRegistryInterface",
    "CheckpointManagerInterface",
    "MetricsManagerInterface",
    "HistoryManagerInterface",
    "EvaluationEngineInterface",
    "ExportPipelineInterface",
    "SchedulerInterface",
    "DatasetLoaderInterface",
    "AugmentationPipelineInterface",
    "HyperparameterManagerInterface",
    "TrainingValidatorInterface",
    "ExperimentManager",
    "ModelRegistry",
    "CheckpointManager",
    "MetricsManager",
    "HistoryManager",
    "EvaluationEngine",
    "ExportPipeline",
    "Callback",
    "CallbackRunner",
    "Scheduler",
    "EarlyStopping",
    "Trainer",
    "TrainingService",
    "TrainingBackendRegistry",
    "YOLOTrainingBackend",
    "DatasetLoader",
    "AugmentationPipeline",
    "TransformDescriptor",
    "HyperparameterManager",
    "TrainingValidator",
    "TrainingPipeline",
]
