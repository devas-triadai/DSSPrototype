"""Evaluation engine — interfaces for model validation, testing, and benchmarking.

Provides the evaluation architecture without model-specific logic.
Concrete evaluation metrics are injected by framework-specific implementations.
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.interfaces import EvaluationEngineInterface
from backend.training.models import EvaluationResult

logger = logging.getLogger("dss.training.evaluator")


class EvaluationEngine(EvaluationEngineInterface):
    """Evaluation engine for model validation, testing, and benchmarking.

    This is an architectural scaffold. Concrete metric computation
    (e.g., COCO mAP, confusion matrix) is implemented by subclasses
    or injected evaluators when a real model is available.
    """

    def __init__(self, reports_dir: Path | None = None) -> None:
        self._config = training_config
        self._reports_dir = reports_dir or self._config.reports_dir
        self._reports_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self, experiment_id: str, checkpoint_path: str, dataset_version: str = "",
    ) -> EvaluationResult:
        logger.info("Validation started: %s checkpoint=%s", experiment_id, checkpoint_path)
        result = EvaluationResult(
            experiment_id=experiment_id,
            checkpoint_path=checkpoint_path,
            dataset_version=dataset_version,
            split="validation",
        )
        self._persist(result)
        logger.info("Validation completed: %s", experiment_id)
        return result

    def test(
        self, experiment_id: str, checkpoint_path: str, dataset_version: str = "",
    ) -> EvaluationResult:
        logger.info("Test evaluation started: %s", experiment_id)
        result = EvaluationResult(
            experiment_id=experiment_id,
            checkpoint_path=checkpoint_path,
            dataset_version=dataset_version,
            split="test",
        )
        self._persist(result)
        logger.info("Test evaluation completed: %s", experiment_id)
        return result

    def benchmark(
        self, experiment_id: str, checkpoint_path: str,
    ) -> EvaluationResult:
        logger.info("Benchmark started: %s", experiment_id)
        result = EvaluationResult(
            experiment_id=experiment_id,
            checkpoint_path=checkpoint_path,
            split="benchmark",
        )
        self._persist(result)
        logger.info("Benchmark completed: %s", experiment_id)
        return result

    def _result_path(self, experiment_id: str, split: str) -> Path:
        return self._reports_dir / f"{experiment_id}_{split}_eval.json"

    def _persist(self, result: EvaluationResult) -> None:
        path = self._result_path(result.experiment_id, result.split)
        path.write_text(json.dumps(result.model_dump(), indent=2, default=str))
