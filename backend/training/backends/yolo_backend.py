"""YOLO training backend — wraps Ultralytics YOLO for CV model training.

This backend is a plugin: the Trainer never imports ultralytics directly.
All YOLO types are encapsulated within this module.

Supports:
  - Single-epoch training steps
  - Validation with COCO metrics
  - Export to PyTorch, ONNX, TorchScript, OpenVINO
  - Checkpoint save/load with resume
  - Per-class metrics and confusion matrix
"""

import logging
from pathlib import Path
from typing import Any

from backend.training.interfaces import TrainingBackendInterface
from backend.training.models import EvaluationResult, TrainingConfigData

logger = logging.getLogger("dss.training.backends.yolo")

YOLO: Any = None
ConfusionMatrix: Any = None
_HAS_ULTRALYTICS = False

try:
    from ultralytics import YOLO as _YOLO  # type: ignore[attr-defined]
    from ultralytics.utils.metrics import ConfusionMatrix as _ConfusionMatrix
    YOLO = _YOLO
    ConfusionMatrix = _ConfusionMatrix
    _HAS_ULTRALYTICS = True
except ImportError:
    pass


class YOLOTrainingBackend(TrainingBackendInterface):
    """Training backend for Ultralytics YOLO models.

    All YOLO types are encapsulated. The external world sees only
    the strongly typed TrainingConfig, EvaluationResult, etc.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._config: TrainingConfigData | None = None
        self._experiment_id: str = ""
        self._dataset_path: str = ""
        self._epochs_completed: int = 0
        self._target_epochs: int = 0
        self._results: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        config: TrainingConfigData,
        experiment_id: str,
        dataset_path: str,
    ) -> None:
        if not _HAS_ULTRALYTICS:
            raise ImportError(
                "ultralytics is not installed. Install it with: pip install ultralytics",
            )

        logger.info(
            "YOLO backend initializing: model=%s, experiment=%s, dataset=%s",
            config.model_name, experiment_id, dataset_path,
        )

        self._config = config
        self._experiment_id = experiment_id
        self._dataset_path = dataset_path
        self._epochs_completed = 0
        self._target_epochs = config.epochs

        model_name = self._resolve_model_name(config.model_name)
        self._model = YOLO(model_name)

        logger.info("YOLO backend initialized: %s", model_name)

    def shutdown(self) -> None:
        """Release model and clear GPU memory."""
        if self._model is not None:
            try:
                import gc

                import torch

                del self._model
                torch.cuda.empty_cache()
                gc.collect()
            except Exception:
                pass
            self._model = None
        logger.info("YOLO backend shut down")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int, learning_rate: float) -> dict[str, float]:
        """Run one training epoch using YOLO's resume mechanism.

        Each call increments the target total epochs by 1, causing
        YOLO to train exactly one more epoch from the last checkpoint.
        """
        if self._model is None:
            raise RuntimeError("YOLO backend not initialized. Call initialize() first.")

        logger.debug("YOLO train_epoch: epoch=%d, lr=%.6f", epoch, learning_rate)

        target = self._epochs_completed + 1
        result = self._model.train(
            data=self._dataset_path,
            epochs=target,
            batch=self._config.batch_size if self._config else 16,
            lr0=learning_rate,
            imgsz=self._config.image_size[0] if self._config else 640,
            device=self._config.device if self._config else "cpu",
            workers=self._config.workers if self._config else 4,
            optimizer=self._config.optimizer if self._config else "Adam",
            seed=self._config.seed if self._config else 42,
            resume=True,
            exist_ok=True,
            verbose=False,
            val=False,
            amp=self._config.mixed_precision if self._config else False,
        )

        self._epochs_completed = target
        self._results = result if isinstance(result, dict) else {}

        metrics = self._extract_train_metrics(result)
        return metrics

    def _extract_train_metrics(self, result: Any) -> dict[str, float]:
        """Extract per-epoch training metrics from YOLO's result object."""
        metrics: dict[str, float] = {
            "training_loss": 0.0,
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "dfl_loss": 0.0,
            "learning_rate": 0.0,
            "epoch_time_ms": 0.0,
        }

        if result is None:
            return metrics

        try:
            if hasattr(result, "box_loss") and result.box_loss is not None:
                metrics["box_loss"] = float(result.box_loss)
            if hasattr(result, "cls_loss") and result.cls_loss is not None:
                metrics["cls_loss"] = float(result.cls_loss)
            if hasattr(result, "dfl_loss") and result.dfl_loss is not None:
                metrics["dfl_loss"] = float(result.dfl_loss)
            if hasattr(result, "lr") and result.lr is not None:
                metrics["learning_rate"] = float(result.lr)

            box = metrics.get("box_loss", 0.0)
            cls = metrics.get("cls_loss", 0.0)
            dfl = metrics.get("dfl_loss", 0.0)
            metrics["training_loss"] = box + cls + dfl
        except Exception:
            pass

        try:
            if hasattr(result, "speed") and result.speed is not None:
                speed = result.speed
                if isinstance(speed, dict):
                    total = sum(float(v) for v in speed.values() if v is not None)
                    metrics["epoch_time_ms"] = total
        except Exception:
            pass

        return metrics

    # ------------------------------------------------------------------
    # Validation / Test
    # ------------------------------------------------------------------

    def validate(
        self,
        checkpoint_path: str,
        dataset_path: str,
        experiment_id: str = "",
    ) -> EvaluationResult:
        if self._model is None:
            raise RuntimeError("YOLO backend not initialized.")

        logger.info("YOLO validation: checkpoint=%s", checkpoint_path)

        try:
            val_model = YOLO(checkpoint_path)
            results = val_model.val(
                data=dataset_path,
                split="val",
                device=self._config.device if self._config else "cpu",
                verbose=False,
            )
        except Exception as e:
            logger.error("YOLO validation failed: %s", e)
            return EvaluationResult(
                experiment_id=experiment_id or self._experiment_id,
                checkpoint_path=checkpoint_path,
                dataset_version="",
                split="validation",
            )

        return self._extract_eval_result(
            results, experiment_id, checkpoint_path, "validation",
        )

    def test(
        self,
        checkpoint_path: str,
        dataset_path: str,
        experiment_id: str = "",
    ) -> EvaluationResult:
        if self._model is None:
            raise RuntimeError("YOLO backend not initialized.")

        logger.info("YOLO test: checkpoint=%s", checkpoint_path)

        try:
            test_model = YOLO(checkpoint_path)
            results = test_model.val(
                data=dataset_path,
                split="test",
                device=self._config.device if self._config else "cpu",
                verbose=False,
            )
        except Exception as e:
            logger.error("YOLO test failed: %s", e)
            return EvaluationResult(
                experiment_id=experiment_id or self._experiment_id,
                checkpoint_path=checkpoint_path,
                dataset_version="",
                split="test",
            )

        return self._extract_eval_result(
            results, experiment_id, checkpoint_path, "test",
        )

    def _extract_eval_result(
        self,
        results: Any,
        experiment_id: str,
        checkpoint_path: str,
        split: str,
    ) -> EvaluationResult:
        """Convert YOLO validation results into EvaluationResult."""
        if results is None:
            return EvaluationResult(
                experiment_id=experiment_id or self._experiment_id,
                checkpoint_path=checkpoint_path,
                split=split,
            )

        kwargs: dict[str, Any] = {
            "experiment_id": experiment_id or self._experiment_id,
            "checkpoint_path": checkpoint_path,
            "split": split,
        }

        try:
            if hasattr(results, "box"):
                box = results.box
                if box is not None:
                    kwargs["mAP50"] = getattr(box, "map50", None)
                    kwargs["mAP50_95"] = getattr(box, "map", None)
                    if hasattr(box, "mp") and box.mp is not None:
                        kwargs["precision"] = float(box.mp)
                    if hasattr(box, "mr") and box.mr is not None:
                        kwargs["recall"] = float(box.mr)

            if hasattr(results, "speed"):
                speed = results.speed
                if isinstance(speed, dict):
                    inference = speed.get("inference", None)
                    if inference is not None:
                        kwargs["inference_time_ms"] = float(inference) / 1000.0

            if hasattr(results, "confusion_matrix"):
                cm = results.confusion_matrix
                if cm is not None and hasattr(cm, "matrix"):
                    kwargs["confusion_matrix"] = cm.matrix.tolist()

            if hasattr(results, "ap_class_index"):
                cls_indices = results.ap_class_index
                if cls_indices is not None and hasattr(results, "class_metrics"):
                    per_class: dict[str, dict[str, float]] = {}
                    for idx in cls_indices:
                        cls_name = f"class_{idx}"
                        cls_metrics: dict[str, float] = {}
                        if hasattr(results, "class_metrics") and results.class_metrics:
                            cm_data = results.class_metrics.get(idx, {})
                            if isinstance(cm_data, dict):
                                cls_metrics = {str(k): float(v) for k, v in cm_data.items()}
                        per_class[cls_name] = cls_metrics
                    kwargs["per_class_metrics"] = per_class

            if hasattr(results, "total_images"):
                kwargs["total_images"] = int(results.total_images)

            if "mAP50" in kwargs and kwargs["mAP50"] is not None:
                kwargs["mAP50"] = float(kwargs["mAP50"])
            if "mAP50_95" in kwargs and kwargs["mAP50_95"] is not None:
                kwargs["mAP50_95"] = float(kwargs["mAP50_95"])

        except Exception as e:
            logger.warning("Failed to extract full evaluation metrics: %s", e)

        return EvaluationResult(**kwargs)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(
        self,
        checkpoint_path: str,
        format_name: str,
        output_dir: Path,
        model_id: str = "",
        experiment_id: str = "",
    ) -> str:
        """Export a trained YOLO model to the specified format.

        Supported formats: torch, onnx, torchscript, openvino.
        """
        logger.info("YOLO export: format=%s, checkpoint=%s", format_name, checkpoint_path)

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        export_model = YOLO(checkpoint_path)

        format_map: dict[str, str] = {
            "torch": "pt",
            "onnx": "onnx",
            "torchscript": "torchscript",
            "openvino": "openvino",
        }

        ultralytics_format = format_map.get(format_name)
        if ultralytics_format is None:
            raise ValueError(
                f"Unsupported export format: '{format_name}'. "
                f"Supported: {list(format_map.keys())}",
            )

        exported_path = export_model.export(
            format=ultralytics_format,
            imgsz=self._config.image_size[0] if self._config else 640,
            device=self._config.device if self._config else "cpu",
            verbose=False,
        )

        import shutil
        if exported_path and Path(exported_path).exists():
            dest = output_dir / Path(exported_path).name
            shutil.copy2(str(exported_path), str(dest))
            result_path = str(dest)
        else:
            result_path = str(output_dir / f"model.{ultralytics_format}")

        logger.info("YOLO export completed: %s → %s", format_name, result_path)
        return result_path

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_path: str, epoch: int) -> dict[str, object]:
        """Save current model weights to a file.

        YOLO automatically saves checkpoints during training. This
        method copies the last saved weights to the specified path.
        """
        if self._model is None:
            raise RuntimeError("YOLO backend not initialized.")

        output_path = str(Path(output_path).resolve())

        last_ckpt = self._find_last_checkpoint()
        if last_ckpt and Path(last_ckpt).exists():
            import shutil
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(last_ckpt), output_path)
            file_size = Path(output_path).stat().st_size
        else:
            self._model.save(output_path)
            file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0

        return {
            "path": output_path,
            "epoch": epoch,
            "file_size_bytes": file_size,
        }

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model weights from a checkpoint file."""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._model = YOLO(checkpoint_path)
        logger.info("YOLO checkpoint loaded: %s", checkpoint_path)

    def resume(self, checkpoint_path: str) -> int:
        """Resume training from a checkpoint.

        Returns the epoch to resume from (last completed epoch).
        """
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self._model = YOLO(checkpoint_path)
        self._epochs_completed = 0
        logger.info("YOLO backend ready to resume from: %s", checkpoint_path)
        return self._epochs_completed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_model_name(self, name: str) -> str:
        """Resolve a short model name to a pretrained YOLO identifier."""
        pretrained: dict[str, str] = {
            "yolov8n": "yolov8n.pt",
            "yolov8s": "yolov8s.pt",
            "yolov8m": "yolov8m.pt",
            "yolov8l": "yolov8l.pt",
            "yolov8x": "yolov8x.pt",
            "yolov9t": "yolov9t.pt",
            "yolov9s": "yolov9s.pt",
            "yolov9m": "yolov9m.pt",
            "yolov9c": "yolov9c.pt",
            "yolov9e": "yolov9e.pt",
            "yolov10n": "yolov10n.pt",
            "yolov10s": "yolov10s.pt",
            "yolov10m": "yolov10m.pt",
            "yolov10l": "yolov10l.pt",
            "yolov10x": "yolov10x.pt",
            "yolo11n": "yolo11n.pt",
            "yolo11s": "yolo11s.pt",
            "yolo11m": "yolo11m.pt",
            "yolo11l": "yolo11l.pt",
            "yolo11x": "yolo11x.pt",
        }
        if name in pretrained:
            return pretrained[name]
        if name.endswith(".pt"):
            return name
        return f"{name}.pt"

    def _find_last_checkpoint(self) -> str | None:
        """Find YOLO's last training checkpoint."""
        if self._config is None:
            return None
        project = "runs/train"
        ckpt_dir = Path(project)
        if not ckpt_dir.exists():
            return None
        exp_dirs = sorted(ckpt_dir.glob("exp*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not exp_dirs:
            return None
        last = exp_dirs[0] / "weights" / "last.pt"
        if last.exists():
            return str(last)
        best = exp_dirs[0] / "weights" / "best.pt"
        if best.exists():
            return str(best)
        return None
