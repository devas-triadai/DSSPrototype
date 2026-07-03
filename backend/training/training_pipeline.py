"""Training pipeline — single orchestrator for the full training lifecycle.

Pipeline stages:
  1. Validate Dataset (production_ready check)
  2. Load Dataset
  3. Augment (build augmentation config)
  4. Validate Config
  5. Train
  6. Evaluate
  7. Checkpoint
  8. Register Model
  9. Export
  10. Record Experiment
"""

import logging
from pathlib import Path

from backend.training.exceptions import (
    DatasetNotReadyError,
    PreValidationError,
)
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
from backend.training.models import (
    AugmentationConfig,
    DatasetLoadResult,
    TrainingConfigData,
    TrainingResult,
)
from backend.training.trainer import Trainer

logger = logging.getLogger("dss.training.training_pipeline")


class TrainingPipeline:
    """Coordinates the full training lifecycle.

    All dependencies are injected via the constructor with sensible defaults.
    """

    def __init__(
        self,
        dataset_loader: DatasetLoaderInterface,
        augmentation_pipeline: AugmentationPipelineInterface,
        hyperparameter_manager: HyperparameterManagerInterface,
        validator: TrainingValidatorInterface,
        experiment_manager: ExperimentManagerInterface,
        model_registry: ModelRegistryInterface,
        checkpoint_manager: CheckpointManagerInterface,
        metrics_manager: MetricsManagerInterface,
        history_manager: HistoryManagerInterface,
        evaluation_engine: EvaluationEngineInterface,
        export_pipeline: ExportPipelineInterface,
        training_backend: TrainingBackendInterface,
        scheduler: SchedulerInterface | None = None,
    ) -> None:
        self._dataset_loader = dataset_loader
        self._augmentation_pipeline = augmentation_pipeline
        self._hyperparameter_manager = hyperparameter_manager
        self._validator = validator
        self._experiment_manager = experiment_manager
        self._model_registry = model_registry
        self._checkpoint_manager = checkpoint_manager
        self._metrics_manager = metrics_manager
        self._history_manager = history_manager
        self._evaluation_engine = evaluation_engine
        self._export_pipeline = export_pipeline
        self._training_backend = training_backend
        self._scheduler = scheduler

    def run(
        self,
        config: TrainingConfigData,
        augmentation_config: AugmentationConfig | None = None,
        dataset_metadata: object | None = None,
    ) -> TrainingResult:
        logger.info(
            "Pipeline started: %s (model=%s)",
            config.experiment_name, config.model_name,
        )

        stage = "validate_dataset"
        try:
            # ── Stage 1: Validate Dataset ──────────────────────────────────
            stage = "validate_dataset"
            dataset = self._dataset_loader.load_dataset(
                dataset_name=config.model_name,
                dataset_version=config.dataset_version,
            )
            logger.info("Stage 1/10 passed: dataset validation")

            # ── Stage 2: Load Dataset ──────────────────────────────────────
            stage = "load_dataset"
            dataset_path = self._resolve_dataset_path(dataset, config)
            logger.info("Stage 2/10 passed: dataset loaded from %s", dataset_path)

            # ── Stage 3: Augment ─────────────────────────────────────────────
            stage = "augment"
            aug_cfg = augmentation_config or AugmentationConfig()
            self._augmentation_pipeline.create_pipeline(aug_cfg)
            logger.info(
                "Stage 3/10 passed: augmentation pipeline '%s' created",
                aug_cfg.name,
            )

            # ── Stage 4: Validate Config ────────────────────────────────────
            stage = "validate_config"
            validation = self._validator.validate_config(config)
            if not validation.passed:
                raise PreValidationError(
                    f"Configuration validation failed: {'; '.join(validation.errors)}",
                )
            logger.info(
                "Stage 4/10 passed: config validation (%d warnings)", validation.num_warnings,
            )

            # ── Stage 5-10: Train → Evaluate → Checkpoint → Register → Export
            stage = "train"
            trainer = self._create_trainer()
            result = trainer.train(config, dataset_metadata)

            logger.info(
                "Pipeline completed: %s (%d epochs, status=%s)",
                result.experiment_id, result.total_epochs_completed, result.status,
            )
            return result

        except DatasetNotReadyError:
            logger.error("Pipeline failed at stage '%s': dataset not ready", stage)
            raise
        except PreValidationError:
            logger.error("Pipeline failed at stage '%s': validation error", stage)
            raise
        except Exception:
            logger.error("Pipeline failed at stage '%s'", stage)
            raise

    def _create_trainer(self) -> Trainer:
        return Trainer(
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

    def _resolve_dataset_path(
        self, dataset: DatasetLoadResult, config: TrainingConfigData,
    ) -> str:
        if dataset.train_path and Path(dataset.train_path).exists():
            return dataset.train_path
        yaml_path = Path("datasets") / "exports" / "yolo" / config.dataset_version / "data.yaml"
        if yaml_path.exists():
            return str(yaml_path)
        return dataset.train_path or str(
            Path("datasets") / "exports" / "yolo" / config.dataset_version / "data.yaml",
        )
