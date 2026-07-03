"""Strongly typed Pydantic models for the Training Platform.

Every model uses frozen=True for immutability, following the DSS contract pattern.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfigData(BaseModel):
    """Complete training configuration.

    Captures every hyperparameter needed to reproduce a training run.
    """

    model_config = ConfigDict(frozen=True)

    model_name: str = Field(...)
    model_version: str = Field(default="1.0.0")
    dataset_version: str = Field(default="1.0.0")
    experiment_name: str = Field(default="default")
    batch_size: int = Field(default=16)
    epochs: int = Field(default=100)
    learning_rate: float = Field(default=0.001)
    optimizer: str = Field(default="adam")
    scheduler: str = Field(default="cosine")
    weight_decay: float = Field(default=0.0001)
    image_size: tuple[int, int] = Field(default=(640, 640))
    augmentation: str = Field(default="default")
    mixed_precision: bool = Field(default=False)
    device: str = Field(default="cpu")
    workers: int = Field(default=4)
    seed: int = Field(default=42)
    resume_checkpoint: str | None = Field(default=None)
    early_stopping_patience: int | None = Field(default=None)
    early_stopping_delta: float = Field(default=0.001)
    validation_interval: int = Field(default=1)
    save_interval: int = Field(default=5)
    notes: str = Field(default="")

    def validate_config(self) -> list[str]:
        errors: list[str] = []
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.epochs < 1:
            errors.append("epochs must be >= 1")
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        if self.validation_interval < 1:
            errors.append("validation_interval must be >= 1")
        if self.save_interval < 1:
            errors.append("save_interval must be >= 1")
        return errors


class ExperimentData(BaseModel):
    """Tracks a single training experiment."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    experiment_name: str = Field(...)
    dataset_version: str = Field(default="")
    config: TrainingConfigData = Field(...)
    training_start: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    training_end: str | None = Field(default=None)
    duration_seconds: float | None = Field(default=None)
    best_epoch: int | None = Field(default=None)
    best_metric: float | None = Field(default=None)
    best_metric_name: str = Field(default="")
    status: str = Field(
        default="created", description="created | running | completed | failed | interrupted",
    )
    notes: str = Field(default="")


class ModelEntry(BaseModel):
    """Entry in the model registry."""

    model_config = ConfigDict(frozen=True)

    model_id: str = Field(...)
    model_name: str = Field(...)
    architecture: str = Field(default="")
    version: str = Field(default="1.0.0")
    dataset_version: str = Field(default="")
    experiment_id: str = Field(default="")
    checkpoint_path: str = Field(default="")
    metrics: dict[str, float] = Field(default_factory=dict)
    created_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    training_status: str = Field(
        default="pending", description="pending | training | completed | failed",
    )
    framework: str = Field(default="pytorch")
    export_formats: list[str] = Field(default_factory=list)
    checksum: str = Field(default="")


class CheckpointData(BaseModel):
    """Metadata for a saved checkpoint."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    epoch: int = Field(...)
    path: str = Field(...)
    metric_value: float | None = Field(default=None)
    is_best: bool = Field(default=False)
    is_latest: bool = Field(default=False)
    file_size_bytes: int = Field(default=0)
    checksum: str = Field(default="")
    saved_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, object] = Field(default_factory=dict)


class MetricData(BaseModel):
    """A single metric snapshot during training."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    epoch: int = Field(...)
    step: int = Field(default=0)
    training_loss: float | None = Field(default=None)
    validation_loss: float | None = Field(default=None)
    precision: float | None = Field(default=None)
    recall: float | None = Field(default=None)
    mAP50: float | None = Field(default=None)  # noqa: N815
    mAP50_95: float | None = Field(default=None)  # noqa: N815
    classification_accuracy: float | None = Field(default=None)
    learning_rate: float | None = Field(default=None)
    epoch_time_ms: float | None = Field(default=None)
    gpu_memory_mb: float | None = Field(default=None)
    cpu_memory_mb: float | None = Field(default=None)
    throughput_items_per_sec: float | None = Field(default=None)
    additional_metrics: dict[str, float] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class HistoryEntry(BaseModel):
    """A single row in training history."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    epoch: int = Field(...)
    loss: float = Field(default=0.0)
    metrics: dict[str, float] = Field(default_factory=dict)
    learning_rate: float = Field(default=0.0)
    checkpoint_path: str = Field(default="")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvaluationResult(BaseModel):
    """Result of a model evaluation run."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    checkpoint_path: str = Field(...)
    dataset_version: str = Field(default="")
    split: str = Field(default="validation", description="validation | test | benchmark")
    precision: float | None = Field(default=None)
    recall: float | None = Field(default=None)
    mAP50: float | None = Field(default=None)  # noqa: N815
    mAP50_95: float | None = Field(default=None)  # noqa: N815
    classification_accuracy: float | None = Field(default=None)
    per_class_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = Field(default=None)
    inference_time_ms: float | None = Field(default=None)
    total_images: int = Field(default=0)
    additional_metrics: dict[str, float] = Field(default_factory=dict)
    evaluated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class ExportData(BaseModel):
    """Metadata for a model export."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    model_id: str = Field(...)
    format_name: str = Field(...)
    output_path: str = Field(...)
    file_size_bytes: int = Field(default=0)
    checksum: str = Field(default="")
    framework_version: str = Field(default="")
    opset_version: int | None = Field(default=None)
    input_shape: tuple[int, ...] | None = Field(default=None)
    output_shape: tuple[int, ...] | None = Field(default=None)
    exported_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class SchedulerConfig(BaseModel):
    """Configuration for a learning rate scheduler."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="cosine")
    base_lr: float = Field(default=0.001)
    min_lr: float = Field(default=1e-6)
    warmup_epochs: int = Field(default=0)
    warmup_start_lr: float = Field(default=1e-6)
    params: dict[str, float] = Field(default_factory=dict)


class EarlyStoppingConfig(BaseModel):
    """Configuration for early stopping."""

    model_config = ConfigDict(frozen=True)

    patience: int = Field(default=10)
    min_delta: float = Field(default=0.001)
    monitor: str = Field(default="validation_loss")
    mode: str = Field(default="min", description="min | max")
    restore_best_checkpoint: bool = Field(default=True)


class DatasetLoadResult(BaseModel):
    """Result of loading a dataset for training."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(...)
    dataset_version: str = Field(...)
    quality_report_version: str = Field(default="")
    production_ready: bool = Field(...)
    train_path: str = Field(default="")
    val_path: str = Field(default="")
    test_path: str = Field(default="")
    class_names: list[str] = Field(default_factory=list)
    num_classes: int = Field(default=0, ge=0)
    total_images: int = Field(default=0, ge=0)
    total_annotations: int = Field(default=0, ge=0)


class AugmentationConfig(BaseModel):
    """Configuration for an augmentation pipeline."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(default="default")
    flip_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    rotate_degrees: tuple[float, float] = Field(default=(-10.0, 10.0))
    rotate_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    scale_min: float = Field(default=0.5, ge=0.0)
    scale_max: float = Field(default=2.0, ge=0.0)
    scale_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    crop_min: float = Field(default=0.5, ge=0.0, le=1.0)
    crop_max: float = Field(default=1.0, ge=0.0, le=1.0)
    crop_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    brightness_range: tuple[float, float] = Field(default=(0.8, 1.2))
    brightness_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    contrast_range: tuple[float, float] = Field(default=(0.8, 1.2))
    contrast_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    blur_kernel_size: int = Field(default=5, ge=1)
    blur_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    noise_intensity: float = Field(default=0.05, ge=0.0, le=1.0)
    noise_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    mosaic_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    mixup_probability: float = Field(default=0.5, ge=0.0, le=1.0)


class HyperparameterProfile(BaseModel):
    """A named hyperparameter profile for training."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(...)
    learning_rate: float = Field(default=0.001, gt=0.0)
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=100, ge=1)
    optimizer: str = Field(default="adam")
    scheduler: str = Field(default="cosine")
    image_size: tuple[int, int] = Field(default=(640, 640))
    weight_decay: float = Field(default=0.0001, ge=0.0)
    seed: int = Field(default=42)
    early_stopping_patience: int | None = Field(default=None)
    early_stopping_delta: float = Field(default=0.001, ge=0.0)
    mixed_precision: bool = Field(default=False)
    warmup_epochs: int = Field(default=0, ge=0)
    warmup_start_lr: float = Field(default=1e-6)
    min_lr: float = Field(default=1e-6)
    description: str = Field(default="")


class PreValidationResult(BaseModel):
    """Result of pre-training configuration validation."""

    model_config = ConfigDict(frozen=True)

    passed: bool = Field(...)
    errors: tuple[str, ...] = Field(default_factory=tuple)
    warnings: tuple[str, ...] = Field(default_factory=tuple)
    dataset_ready: bool = Field(default=False)
    config_valid: bool = Field(default=False)
    augmentation_valid: bool = Field(default=False)
    num_errors: int = Field(default=0, ge=0)
    num_warnings: int = Field(default=0, ge=0)


class TrainingResult(BaseModel):
    """Complete result of a training run."""

    model_config = ConfigDict(frozen=True)

    experiment_id: str = Field(...)
    model_id: str = Field(...)
    total_epochs_completed: int = Field(default=0)
    best_epoch: int | None = Field(default=None)
    best_metric: float | None = Field(default=None)
    best_metric_name: str = Field(default="")
    training_duration_seconds: float = Field(default=0.0)
    final_metrics: MetricData | None = Field(default=None)
    status: str = Field(default="completed", description="completed | interrupted | failed")
    export_results: list[ExportData] = Field(default_factory=list)
    evaluation_result: EvaluationResult | None = Field(default=None)
    completed_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
