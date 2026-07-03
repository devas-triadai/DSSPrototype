"""Trainer — coordinates the full training lifecycle through DI.

The Trainer depends ONLY on interfaces. It orchestrates:
  Dataset → Backend → Experiment → Checkpoint → Metrics → Callbacks
  → History → Evaluation → Export → Model Registration

No concrete implementations are coupled here.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.training.callbacks import Callback, CallbackRunner
from backend.training.early_stopping import EarlyStopping
from backend.training.exceptions import TrainingInterruptedError
from backend.training.interfaces import (
    CheckpointManagerInterface,
    EvaluationEngineInterface,
    ExperimentManagerInterface,
    ExportPipelineInterface,
    HistoryManagerInterface,
    MetricsManagerInterface,
    ModelRegistryInterface,
    SchedulerInterface,
    TrainerInterface,
    TrainingBackendInterface,
)
from backend.training.models import (
    CheckpointData,
    ExperimentData,
    ExportData,
    HistoryEntry,
    MetricData,
    ModelEntry,
    TrainingConfigData,
    TrainingResult,
)
from backend.training.scheduler import Scheduler

logger = logging.getLogger("dss.training.trainer")


class Trainer(TrainerInterface):
    """Coordinates the full training lifecycle.

    All dependencies are injected via the constructor with sensible defaults.
    """

    def __init__(
        self,
        experiment_manager: ExperimentManagerInterface,
        model_registry: ModelRegistryInterface,
        checkpoint_manager: CheckpointManagerInterface,
        metrics_manager: MetricsManagerInterface,
        history_manager: HistoryManagerInterface,
        evaluation_engine: EvaluationEngineInterface,
        export_pipeline: ExportPipelineInterface,
        training_backend: TrainingBackendInterface,
        scheduler: SchedulerInterface | None = None,
        callbacks: list[Callback] | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        self._experiment_manager = experiment_manager
        self._model_registry = model_registry
        self._checkpoint_manager = checkpoint_manager
        self._metrics_manager = metrics_manager
        self._history_manager = history_manager
        self._evaluation_engine = evaluation_engine
        self._export_pipeline = export_pipeline
        self._training_backend = training_backend
        self._scheduler = scheduler or Scheduler()
        self._callback_runner = CallbackRunner(callbacks or [])
        self._early_stopping = early_stopping

    def add_callback(self, callback: Callback) -> None:
        self._callback_runner.add_callback(callback)

    def train(
        self,
        config: TrainingConfigData,
        dataset_metadata: object | None = None,
    ) -> TrainingResult:
        logger.info(
            "Training started: %s (model=%s)", config.experiment_name, config.model_name,
        )

        errors = config.validate_config()
        if errors:
            for e in errors:
                logger.error("Configuration error: %s", e)

        # ── Resolve dataset path ──────────────────────────────────────────
        dataset_path = self._resolve_dataset_path(config, dataset_metadata)
        logger.info("Dataset path resolved: %s", dataset_path)

        # ── Experiment lifecycle ───────────────────────────────────────────
        experiment = self._experiment_manager.create_experiment(config)
        experiment = self._experiment_manager.update_experiment(
            _with_status(experiment, "running"),
        )

        # ── Model registration ─────────────────────────────────────────────
        model_entry = self._model_registry.register_model(
            ModelEntry(
                model_id=_generate_model_id(),
                model_name=config.model_name,
                architecture=config.model_name,
                version=config.model_version,
                dataset_version=config.dataset_version,
                experiment_id=experiment.experiment_id,
                training_status="training",
                framework="pytorch",
            ),
        )

        # ── Early stopping ─────────────────────────────────────────────────
        early_stopping: EarlyStopping | None = None
        if config.early_stopping_patience is not None and config.early_stopping_patience > 0:
            early_stopping = self._early_stopping or EarlyStopping(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_delta,
                monitor="mAP50",
                mode="max",
                restore_best_checkpoint=True,
            )

        # ── Backend initialization ─────────────────────────────────────────
        self._training_backend.initialize(
            config=config,
            experiment_id=experiment.experiment_id,
            dataset_path=dataset_path,
        )

        training_start = time.monotonic()

        self._callback_runner.on_train_start()

        best_metric: float | None = None
        best_epoch: int | None = None
        total_epochs_completed = 0
        last_checkpoint_path: str = ""

        try:
            for epoch in range(1, config.epochs + 1):
                epoch_start = time.monotonic()
                self._callback_runner.on_epoch_start(epoch)

                current_lr = self._scheduler.get_lr(epoch, config.learning_rate)

                # ── Train one epoch through the backend ────────────────────
                epoch_metrics = self._training_backend.train_epoch(
                    epoch=epoch,
                    learning_rate=current_lr,
                )

                epoch_time_ms = (time.monotonic() - epoch_start) * 1000

                training_loss = epoch_metrics.get("training_loss", 0.0)
                box_loss = epoch_metrics.get("box_loss")
                cls_loss = epoch_metrics.get("cls_loss")
                dfl_loss = epoch_metrics.get("dfl_loss")
                reported_lr = epoch_metrics.get("learning_rate", current_lr)
                reported_epoch_time = epoch_metrics.get("epoch_time_ms", epoch_time_ms)

                # ── Record metrics ─────────────────────────────────────────
                metric_kwargs: dict[str, Any] = {
                    "experiment_id": experiment.experiment_id,
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "learning_rate": reported_lr,
                    "epoch_time_ms": reported_epoch_time,
                }
                if box_loss is not None:
                    metric_kwargs["additional_metrics"] = {
                        "box_loss": box_loss,
                        "cls_loss": cls_loss or 0.0,
                        "dfl_loss": dfl_loss or 0.0,
                    }
                metric = MetricData(**metric_kwargs)
                self._metrics_manager.record(metric)

                # ── Save checkpoint ────────────────────────────────────────
                checkpoint: CheckpointData | None = None
                if epoch % config.save_interval == 0 or epoch == 1:
                    ckpt_dir = self._checkpoint_manager.checkpoints_dir / experiment.experiment_id
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = str(ckpt_dir / f"epoch_{epoch:04d}.pt")

                    self._training_backend.save_checkpoint(ckpt_path, epoch)

                    checkpoint = self._checkpoint_manager.save_checkpoint(
                        experiment_id=experiment.experiment_id,
                        epoch=epoch,
                        metric_value=training_loss,
                        metadata={
                            "box_loss": box_loss,
                            "cls_loss": cls_loss,
                            "dfl_loss": dfl_loss,
                            "learning_rate": reported_lr,
                            "epoch_time_ms": reported_epoch_time,
                        },
                    )

                    if Path(ckpt_path).exists():
                        file_size = Path(ckpt_path).stat().st_size
                        checkpoint = CheckpointData(
                            experiment_id=checkpoint.experiment_id,
                            epoch=checkpoint.epoch,
                            path=checkpoint.path,
                            metric_value=checkpoint.metric_value,
                            is_best=checkpoint.is_best,
                            is_latest=checkpoint.is_latest,
                            file_size_bytes=file_size,
                            checksum=checkpoint.checksum,
                            saved_at=checkpoint.saved_at,
                            metadata=checkpoint.metadata,
                        )

                    last_checkpoint_path = ckpt_path
                    self._callback_runner.on_checkpoint_saved(checkpoint)

                # ── Validation ─────────────────────────────────────────────
                val_loss: float | None = None
                val_precision: float | None = None
                val_recall: float | None = None
                val_map50: float | None = None
                val_map50_95: float | None = None

                if epoch % config.validation_interval == 0 and last_checkpoint_path:
                    val_result = self._training_backend.validate(
                        checkpoint_path=last_checkpoint_path,
                        dataset_path=dataset_path,
                        experiment_id=experiment.experiment_id,
                    )

                    val_loss = training_loss
                    val_precision = val_result.precision
                    val_recall = val_result.recall
                    val_map50 = val_result.mAP50
                    val_map50_95 = val_result.mAP50_95

                    if val_map50 is not None:
                        if best_metric is None or val_map50 > best_metric:
                            best_metric = val_map50
                            best_epoch = epoch

                    # ── Update metric with validation results ─────────────
                    updated_metric = MetricData(
                        experiment_id=experiment.experiment_id,
                        epoch=epoch,
                        training_loss=training_loss,
                        validation_loss=val_loss,
                        precision=val_precision,
                        recall=val_recall,
                        mAP50=val_map50,
                        mAP50_95=val_map50_95,
                        learning_rate=reported_lr,
                        epoch_time_ms=reported_epoch_time,
                        additional_metrics=dict(metric_kwargs.get("additional_metrics", {})),
                    )
                    self._metrics_manager.record(updated_metric)

                    # ── Evaluation engine persistence ──────────────────────
                    self._evaluation_engine.validate(
                        experiment_id=experiment.experiment_id,
                        checkpoint_path=last_checkpoint_path,
                        dataset_version=config.dataset_version,
                    )

                    self._callback_runner.on_validation_end(updated_metric)

                # ── History ────────────────────────────────────────────────
                history_metrics: dict[str, float] = {
                    "training_loss": training_loss,
                }
                if val_loss is not None:
                    history_metrics["validation_loss"] = val_loss
                if val_map50 is not None:
                    history_metrics["mAP50"] = val_map50
                if val_precision is not None:
                    history_metrics["precision"] = val_precision
                if val_recall is not None:
                    history_metrics["recall"] = val_recall

                history_entry = HistoryEntry(
                    experiment_id=experiment.experiment_id,
                    epoch=epoch,
                    loss=training_loss,
                    metrics=history_metrics,
                    learning_rate=reported_lr,
                    checkpoint_path=last_checkpoint_path,
                )
                self._history_manager.record_entry(history_entry)
                self._callback_runner.on_epoch_end(epoch, history_entry)

                total_epochs_completed = epoch

                # ── Early stopping check ───────────────────────────────────
                if early_stopping is not None and val_map50 is not None:
                    if early_stopping.check(val_map50, epoch):
                        logger.info(
                            "Early stopping triggered at epoch %d (best mAP50=%.4f)",
                            epoch, early_stopping.best_value or 0.0,
                        )
                        break

            training_duration = time.monotonic() - training_start

            # ── Final checkpoint (best) ────────────────────────────────────
            if best_epoch and last_checkpoint_path:
                model_ckpt_dir = self._checkpoint_manager.checkpoints_dir / experiment.experiment_id
                best_ckpt_path = str(model_ckpt_dir / "best.pt")
                self._training_backend.save_checkpoint(best_ckpt_path, best_epoch)

            # ── Export pipeline ────────────────────────────────────────────
            export_results: list[ExportData] = []
            if last_checkpoint_path:
                export_formats = ["torch", "onnx", "torchscript"]
                for fmt in export_formats:
                    try:
                        self._training_backend.export(
                            checkpoint_path=last_checkpoint_path,
                            format_name=fmt,
                            output_dir=Path(config.model_name) / "exports",
                            model_id=model_entry.model_id,
                            experiment_id=experiment.experiment_id,
                        )
                        export_data = self._export_pipeline.export_to_onnx(
                            experiment_id=experiment.experiment_id,
                            model_id=model_entry.model_id,
                            output_dir=Path(config.model_name) / "exports",
                        )
                        self._callback_runner.on_export(export_data)
                        export_results.append(export_data)
                    except Exception as ex:
                        logger.warning("Export to %s failed: %s", fmt, ex)

            # ── Update experiment ──────────────────────────────────────────
            experiment = self._experiment_manager.update_experiment(
                _with_status(
                    _with_best(experiment, best_epoch, best_metric, "mAP50"),
                    "completed",
                    training_duration,
                ),
            )

            # ── Update model registry ──────────────────────────────────────
            model_entry = self._model_registry.update_model(
                _with_model_status(
                    model_entry, "completed",
                    last_checkpoint_path,
                    {"mAP50": best_metric} if best_metric is not None else {},
                ),
            )

            final_metrics = self._metrics_manager.get_latest_metrics(
                experiment.experiment_id,
            )

            self._callback_runner.on_train_end()

            logger.info(
                "Training completed: %s (%d epochs, best=%.4f)",
                experiment.experiment_id, total_epochs_completed,
                best_metric or 0.0,
            )

            # ── Shutdown backend ──────────────────────────────────────────
            self._training_backend.shutdown()

            return TrainingResult(
                experiment_id=experiment.experiment_id,
                model_id=model_entry.model_id,
                total_epochs_completed=total_epochs_completed,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_metric_name="mAP50",
                training_duration_seconds=training_duration,
                final_metrics=final_metrics,
                status="completed",
                export_results=export_results,
            )

        except TrainingInterruptedError:
            training_duration = time.monotonic() - training_start
            self._experiment_manager.update_experiment(
                _with_status(experiment, "interrupted", training_duration),
            )
            self._model_registry.update_model(
                _with_model_status(model_entry, "failed", "", {}),
            )
            self._callback_runner.on_train_end()
            self._training_backend.shutdown()
            return TrainingResult(
                experiment_id=experiment.experiment_id,
                model_id=model_entry.model_id,
                total_epochs_completed=total_epochs_completed,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_metric_name="mAP50",
                training_duration_seconds=training_duration,
                status="interrupted",
            )

        except Exception as e:
            training_duration = time.monotonic() - training_start
            logger.error("Training failed: %s", e)
            self._experiment_manager.update_experiment(
                _with_status(experiment, "failed", training_duration),
            )
            self._model_registry.update_model(
                _with_model_status(model_entry, "failed", "", {}),
            )
            self._callback_runner.on_train_end()
            self._training_backend.shutdown()
            return TrainingResult(
                experiment_id=experiment.experiment_id,
                model_id=model_entry.model_id,
                total_epochs_completed=total_epochs_completed,
                best_epoch=best_epoch,
                best_metric=best_metric,
                best_metric_name="mAP50",
                training_duration_seconds=training_duration,
                status="failed",
            )

    # ------------------------------------------------------------------
    # Dataset resolution
    # ------------------------------------------------------------------

    def _resolve_dataset_path(
        self,
        config: TrainingConfigData,
        dataset_metadata: object | None,
    ) -> str:
        """Resolve the dataset path from config or metadata.

        If dataset_metadata is provided (from DatasetManagementService),
        the dataset YAML path is extracted. Otherwise, a default path
        is constructed from the config.
        """
        if dataset_metadata is not None:
            try:
                if hasattr(dataset_metadata, "splits") and dataset_metadata.splits is not None:
                    from backend.dataset_manager.models import DatasetMetadata
                    if isinstance(dataset_metadata, DatasetMetadata):
                        export_base = Path("datasets") / "exports" / "yolo" / config.dataset_version
                        yaml_path = export_base / "data.yaml"
                        if yaml_path.exists():
                            return str(yaml_path)
            except Exception:
                pass

        yaml_path = Path("datasets") / "exports" / "yolo" / config.dataset_version / "data.yaml"
        if yaml_path.exists():
            return str(yaml_path)

        alt_path = Path("datasets") / "exports" / "yolo" / "data.yaml"
        if alt_path.exists():
            return str(alt_path)

        logger.warning("Dataset YAML not found, using config path: %s", config.dataset_version)
        return str(Path("datasets") / "exports" / "yolo" / config.dataset_version / "data.yaml")


def _with_status(
    experiment: ExperimentData, status: str, duration: float | None = None,
) -> ExperimentData:
    return ExperimentData(
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.experiment_name,
        dataset_version=experiment.dataset_version,
        config=experiment.config,
        training_start=experiment.training_start,
        training_end=(
            datetime.now(timezone.utc).isoformat()
            if status in ("completed", "failed", "interrupted") else None
        ),
        duration_seconds=duration,
        best_epoch=experiment.best_epoch,
        best_metric=experiment.best_metric,
        best_metric_name=experiment.best_metric_name,
        status=status,
        notes=experiment.notes,
    )


def _with_best(
    experiment: ExperimentData, epoch: int | None,
    metric: float | None, name: str,
) -> ExperimentData:
    return ExperimentData(
        experiment_id=experiment.experiment_id,
        experiment_name=experiment.experiment_name,
        dataset_version=experiment.dataset_version,
        config=experiment.config,
        training_start=experiment.training_start,
        training_end=experiment.training_end,
        duration_seconds=experiment.duration_seconds,
        best_epoch=epoch if epoch else experiment.best_epoch,
        best_metric=metric if metric is not None else experiment.best_metric,
        best_metric_name=name,
        status=experiment.status,
        notes=experiment.notes,
    )


def _with_model_status(
    entry: ModelEntry, status: str, checkpoint_path: str,
    metrics: dict[str, float],
) -> ModelEntry:
    return ModelEntry(
        model_id=entry.model_id,
        model_name=entry.model_name,
        architecture=entry.architecture,
        version=entry.version,
        dataset_version=entry.dataset_version,
        experiment_id=entry.experiment_id,
        checkpoint_path=checkpoint_path,
        metrics={**entry.metrics, **metrics},
        training_status=status,
        framework=entry.framework,
        export_formats=entry.export_formats,
        checksum=entry.checksum,
    )


_model_id_counter = 0


def _generate_model_id() -> str:
    global _model_id_counter
    _model_id_counter += 1
    return f"model_{_model_id_counter:04d}"
