"""Test helpers and mock implementations for training tests."""

from pathlib import Path

from backend.training.interfaces import TrainingBackendInterface
from backend.training.models import EvaluationResult, TrainingConfigData


class MockTrainingBackend(TrainingBackendInterface):
    """A mock training backend for testing the Trainer and TrainingService.

    Returns predictable metrics and does not require any deep learning
    framework or GPU.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._config: TrainingConfigData | None = None
        self._experiment_id = ""
        self._dataset_path = ""
        self._current_epoch = 0

    def initialize(
        self,
        config: TrainingConfigData,
        experiment_id: str,
        dataset_path: str,
    ) -> None:
        self._initialized = True
        self._config = config
        self._experiment_id = experiment_id
        self._dataset_path = dataset_path

    def train_epoch(self, epoch: int, learning_rate: float) -> dict[str, float]:
        self._current_epoch = epoch
        return {
            "training_loss": 1.0 / (epoch + 1),
            "box_loss": 0.5 / (epoch + 1),
            "cls_loss": 0.3 / (epoch + 1),
            "dfl_loss": 0.2 / (epoch + 1),
            "learning_rate": learning_rate,
            "epoch_time_ms": 100.0,
        }

    def validate(
        self,
        checkpoint_path: str,
        dataset_path: str,
        experiment_id: str = "",
    ) -> EvaluationResult:
        epoch = self._current_epoch
        return EvaluationResult(
            experiment_id=experiment_id or self._experiment_id,
            checkpoint_path=checkpoint_path,
            dataset_version="",
            split="validation",
            precision=0.85 + epoch * 0.01,
            recall=0.80 + epoch * 0.01,
            mAP50=0.75 + epoch * 0.02,
            mAP50_95=0.50 + epoch * 0.02,
            total_images=100,
        )

    def test(
        self,
        checkpoint_path: str,
        dataset_path: str,
        experiment_id: str = "",
    ) -> EvaluationResult:
        return EvaluationResult(
            experiment_id=experiment_id or self._experiment_id,
            checkpoint_path=checkpoint_path,
            dataset_version="",
            split="test",
            mAP50=0.80,
            mAP50_95=0.55,
            total_images=100,
        )

    def export(
        self,
        checkpoint_path: str,
        format_name: str,
        output_dir: Path,
        model_id: str = "",
        experiment_id: str = "",
    ) -> str:
        out_path = output_dir / f"model.{format_name}"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text("mock_export")
        return str(out_path)

    def save_checkpoint(self, output_path: str, epoch: int) -> dict[str, object]:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(f"mock_checkpoint_epoch_{epoch}")
        return {
            "path": output_path,
            "epoch": epoch,
            "file_size_bytes": Path(output_path).stat().st_size,
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self._initialized = True

    def resume(self, checkpoint_path: str) -> int:
        self._initialized = True
        return 0

    def shutdown(self) -> None:
        self._initialized = False
