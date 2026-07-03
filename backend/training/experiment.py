"""Experiment manager — tracks every training experiment.

Each experiment stores its full configuration, timing, best metrics,
and status. Experiments are persisted as JSON files.
"""

import json
import logging
import uuid
from pathlib import Path

from backend.training.config import training_config
from backend.training.exceptions import ExperimentNotFoundError
from backend.training.interfaces import ExperimentManagerInterface
from backend.training.models import ExperimentData, TrainingConfigData

logger = logging.getLogger("dss.training.experiment")


class ExperimentManager(ExperimentManagerInterface):
    """Manages the lifecycle of training experiments.

    Each experiment is persisted as JSON in the experiments directory.
    """

    def __init__(self, experiments_dir: Path | None = None) -> None:
        self._config = training_config
        self._experiments_dir = experiments_dir or self._config.experiments_dir
        self._experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, config: TrainingConfigData) -> ExperimentData:
        logger.info("Experiment creation started: %s", config.experiment_name)

        experiment = ExperimentData(
            experiment_id=self._generate_id(),
            experiment_name=config.experiment_name,
            dataset_version=config.dataset_version,
            config=config,
        )

        self._persist(experiment)
        logger.info(
            "Experiment created: %s (%s)", experiment.experiment_name, experiment.experiment_id,
        )
        return experiment

    def get_experiment(self, experiment_id: str) -> ExperimentData | None:
        path = self._experiment_path(experiment_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return ExperimentData(**data)
        except Exception:
            return None

    def list_experiments(self) -> list[ExperimentData]:
        experiments: list[ExperimentData] = []
        for path in sorted(self._experiments_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                experiments.append(ExperimentData(**data))
            except Exception:
                pass
        return experiments

    def update_experiment(self, experiment: ExperimentData) -> ExperimentData:
        if not self._experiment_path(experiment.experiment_id).exists():
            raise ExperimentNotFoundError(f"Experiment not found: {experiment.experiment_id}")

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
            status=experiment.status,
            notes=experiment.notes,
        )
        self._persist(updated)
        logger.info(
            "Experiment updated: %s (status=%s)", experiment.experiment_id, experiment.status,
        )
        return updated

    def delete_experiment(self, experiment_id: str) -> bool:
        path = self._experiment_path(experiment_id)
        if not path.exists():
            return False
        path.unlink()
        logger.info("Experiment deleted: %s", experiment_id)
        return True

    def _generate_id(self) -> str:
        return f"exp_{uuid.uuid4().hex[:12]}"

    def _experiment_path(self, experiment_id: str) -> Path:
        return self._experiments_dir / f"{experiment_id}.json"

    def _persist(self, experiment: ExperimentData) -> None:
        path = self._experiment_path(experiment.experiment_id)
        path.write_text(json.dumps(experiment.model_dump(), indent=2, default=str))
