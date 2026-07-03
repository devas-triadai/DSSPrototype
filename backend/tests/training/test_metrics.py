"""Tests for the MetricsManager."""

import json
import tempfile
from pathlib import Path

from backend.training.metrics import MetricsManager
from backend.training.models import MetricData


def _make_manager() -> tuple[MetricsManager, Path]:
    tmp = Path(tempfile.mkdtemp())
    return MetricsManager(metrics_dir=tmp / "metrics"), tmp


def test_record_and_get() -> None:
    mgr, _ = _make_manager()
    m = MetricData(experiment_id="exp_001", epoch=1, training_loss=0.5)
    mgr.record(m)
    metrics = mgr.get_metrics("exp_001")
    assert len(metrics) == 1
    assert metrics[0].training_loss == 0.5


def test_get_by_epoch() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_002", epoch=1, training_loss=0.8))
    mgr.record(MetricData(experiment_id="exp_002", epoch=2, training_loss=0.5))
    epoch_1 = mgr.get_metrics("exp_002", epoch=1)
    assert len(epoch_1) == 1
    assert epoch_1[0].training_loss == 0.8


def test_get_best_metric_min() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_003", epoch=1, validation_loss=0.8))
    mgr.record(MetricData(experiment_id="exp_003", epoch=2, validation_loss=0.5))
    mgr.record(MetricData(experiment_id="exp_003", epoch=3, validation_loss=0.6))
    best = mgr.get_best_metric("exp_003", "validation_loss", "min")
    assert best is not None
    assert best.epoch == 2
    assert best.validation_loss == 0.5


def test_get_best_metric_max() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_004", epoch=1, mAP50=0.7))
    mgr.record(MetricData(experiment_id="exp_004", epoch=2, mAP50=0.9))
    best = mgr.get_best_metric("exp_004", "mAP50", "max")
    assert best is not None
    assert best.epoch == 2


def test_get_latest_metrics() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_005", epoch=1))
    mgr.record(MetricData(experiment_id="exp_005", epoch=2))
    latest = mgr.get_latest_metrics("exp_005")
    assert latest is not None
    assert latest.epoch == 2


def test_get_latest_metrics_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_latest_metrics("empty") is None


def test_get_best_metric_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_best_metric("empty", "validation_loss") is None


def test_record_persists_to_disk() -> None:
    mgr, tmp = _make_manager()
    m = MetricData(experiment_id="exp_persist", epoch=1, training_loss=0.5)
    mgr.record(m)
    files = list((tmp / "metrics").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["training_loss"] == 0.5


def test_get_best_metric_with_none_values() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_none", epoch=1))
    mgr.record(MetricData(experiment_id="exp_none", epoch=2, validation_loss=0.5))
    best = mgr.get_best_metric("exp_none", "validation_loss", "min")
    assert best is not None
    assert best.epoch == 2


def test_get_metrics_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_metrics("empty") == []


def test_get_best_metric_training_loss_max() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_tloss", epoch=1, training_loss=0.8))
    mgr.record(MetricData(experiment_id="exp_tloss", epoch=2, training_loss=0.5))
    mgr.record(MetricData(experiment_id="exp_tloss", epoch=3, training_loss=0.3))
    best = mgr.get_best_metric("exp_tloss", "training_loss", "min")
    assert best is not None
    assert best.epoch == 3


def test_get_best_metric_precision_max() -> None:
    mgr, _ = _make_manager()
    mgr.record(MetricData(experiment_id="exp_prec", epoch=1, precision=0.8))
    mgr.record(MetricData(experiment_id="exp_prec", epoch=2, precision=0.9))
    best = mgr.get_best_metric("exp_prec", "precision", "max")
    assert best is not None
    assert best.epoch == 2



def test_metrics_with_additional_fields() -> None:
    mgr, _ = _make_manager()
    m = MetricData(
        experiment_id="exp_add", epoch=1, training_loss=0.5,
        additional_metrics={"box_loss": 0.3},
    )
    mgr.record(m)
    metrics = mgr.get_metrics("exp_add")
    assert metrics[0].additional_metrics["box_loss"] == 0.3
