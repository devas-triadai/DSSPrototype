"""Metrics manager — records and retrieves training metrics.

Stores per-epoch metric snapshots as JSON files, with query support
for best-value and latest-metrics lookups.
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.interfaces import MetricsManagerInterface
from backend.training.models import MetricData

logger = logging.getLogger("dss.training.metrics")


class MetricsManager(MetricsManagerInterface):
    """Records and queries training metrics.

    Each metric snapshot is persisted as a JSON file keyed by
    experiment_id, epoch, and step.
    """

    def __init__(self, metrics_dir: Path | None = None) -> None:
        self._config = training_config
        self._metrics_dir = metrics_dir or self._config.metrics_dir
        self._metrics_dir.mkdir(parents=True, exist_ok=True)

    def record(self, metric: MetricData) -> MetricData:
        logger.debug("Metric recorded: %s epoch %d step %d",
                      metric.experiment_id, metric.epoch, metric.step)
        self._persist(metric)
        return metric

    def get_metrics(
        self, experiment_id: str, epoch: int | None = None,
    ) -> list[MetricData]:
        metrics: list[MetricData] = []
        for path in sorted(self._metrics_dir.glob(f"{experiment_id}_*.json")):
            try:
                data = json.loads(path.read_text())
                m = MetricData(**data)
                if epoch is None or m.epoch == epoch:
                    metrics.append(m)
            except Exception:
                pass
        return metrics

    def get_best_metric(
        self, experiment_id: str, metric_name: str = "validation_loss",
        mode: str = "min",
    ) -> MetricData | None:
        metrics = self.get_metrics(experiment_id)
        if not metrics:
            return None

        values = []
        for m in metrics:
            val = self._get_metric_value(m, metric_name)
            if val is not None:
                values.append((val, m))

        if not values:
            return None

        if mode == "min":
            best_val, best_m = min(values, key=lambda x: x[0])
        else:
            best_val, best_m = max(values, key=lambda x: x[0])
        return best_m

    def get_latest_metrics(self, experiment_id: str) -> MetricData | None:
        metrics = self.get_metrics(experiment_id)
        return metrics[-1] if metrics else None

    def _get_metric_value(self, metric: MetricData, name: str) -> float | None:
        val = getattr(metric, name, None)
        if isinstance(val, (int, float)):
            return float(val)
        return None

    def _metric_path(self, metric: MetricData) -> Path:
        return (
            self._metrics_dir
            / f"{metric.experiment_id}_epoch{metric.epoch:04d}_step{metric.step:04d}.json"
        )

    def _persist(self, metric: MetricData) -> None:
        path = self._metric_path(metric)
        path.write_text(json.dumps(metric.model_dump(), indent=2, default=str))
