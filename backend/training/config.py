"""Training Platform configuration.

All values are overridable via environment variables prefixed with ``TR_``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings


class TrainingConfig(BaseSettings):
    """Configuration for the Training Platform.

    Controls training paths, default hyperparameters, and resource allocation.
    """

    model_config = {"env_prefix": "TR_"}

    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    training_root: Path = base_dir / "training"
    experiments_dir: Path = training_root / "experiments"
    models_dir: Path = training_root / "models"
    checkpoints_dir: Path = training_root / "checkpoints"
    metrics_dir: Path = training_root / "metrics"
    logs_dir: Path = training_root / "logs"
    exports_dir: Path = training_root / "exports"
    reports_dir: Path = training_root / "reports"
    history_dir: Path = training_root / "history"
    configs_dir: Path = training_root / "configs"

    default_batch_size: int = 16
    default_epochs: int = 100
    default_learning_rate: float = 0.001
    default_optimizer: str = "adam"
    default_image_size: tuple[int, int] = (640, 640)
    default_seed: int = 42
    default_workers: int = 4
    default_mixed_precision: bool = False
    default_device: str = "cpu"

    # Checkpoint settings
    save_interval: int = 5
    validation_interval: int = 1
    keep_last_n_checkpoints: int = 3

    # Early stopping defaults
    early_stopping_patience: int = 10
    early_stopping_delta: float = 0.001

    # Export settings
    export_onnx_opset: int = 17
    export_torchscript_method: str = "trace"


training_config = TrainingConfig()
