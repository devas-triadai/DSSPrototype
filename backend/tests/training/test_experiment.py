"""Tests for the ExperimentManager."""

import json
import tempfile
from pathlib import Path

import pytest

from backend.training.exceptions import ExperimentNotFoundError
from backend.training.experiment import ExperimentManager
from backend.training.models import ExperimentData, TrainingConfigData


def _make_manager() -> tuple[ExperimentManager, Path]:
    tmp = Path(tempfile.mkdtemp())
    return ExperimentManager(experiments_dir=tmp / "experiments"), tmp


def test_create_experiment() -> None:
    mgr, _ = _make_manager()
    cfg = TrainingConfigData(model_name="yolo", experiment_name="my_exp")
    exp = mgr.create_experiment(cfg)
    assert exp.experiment_name == "my_exp"
    assert exp.status == "created"
    assert exp.experiment_id.startswith("exp_")
    assert exp.config.model_name == "yolo"


def test_get_experiment() -> None:
    mgr, _ = _make_manager()
    cfg = TrainingConfigData(model_name="test", experiment_name="get_test")
    exp = mgr.create_experiment(cfg)
    retrieved = mgr.get_experiment(exp.experiment_id)
    assert retrieved is not None
    assert retrieved.experiment_id == exp.experiment_id


def test_get_experiment_not_found() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_experiment("nonexistent") is None


def test_list_experiments() -> None:
    mgr, _ = _make_manager()
    mgr.create_experiment(TrainingConfigData(model_name="a", experiment_name="exp_a"))
    mgr.create_experiment(TrainingConfigData(model_name="b", experiment_name="exp_b"))
    exps = mgr.list_experiments()
    assert len(exps) == 2


def test_update_experiment() -> None:
    mgr, _ = _make_manager()
    cfg = TrainingConfigData(model_name="test", experiment_name="update_test")
    exp = mgr.create_experiment(cfg)
    updated = mgr.update_experiment(
        ExperimentData(
            experiment_id=exp.experiment_id,
            experiment_name=exp.experiment_name,
            config=exp.config,
            status="running",
        ),
    )
    assert updated.status == "running"


def test_update_experiment_not_found_raises() -> None:
    mgr, _ = _make_manager()
    cfg = TrainingConfigData(model_name="test")
    exp = ExperimentData(experiment_id="nonexistent", experiment_name="test", config=cfg)
    with pytest.raises(ExperimentNotFoundError):
        mgr.update_experiment(exp)


def test_delete_experiment() -> None:
    mgr, _ = _make_manager()
    cfg = TrainingConfigData(model_name="test", experiment_name="delete_test")
    exp = mgr.create_experiment(cfg)
    assert mgr.delete_experiment(exp.experiment_id) is True
    assert mgr.get_experiment(exp.experiment_id) is None


def test_delete_nonexistent_experiment() -> None:
    mgr, _ = _make_manager()
    assert mgr.delete_experiment("nonexistent") is False


def test_create_experiment_persists_to_disk() -> None:
    mgr, tmp = _make_manager()
    cfg = TrainingConfigData(model_name="yolo", experiment_name="persist_test")
    exp = mgr.create_experiment(cfg)
    exp_file = tmp / "experiments" / f"{exp.experiment_id}.json"
    assert exp_file.exists()
    data = json.loads(exp_file.read_text())
    assert data["experiment_name"] == "persist_test"


def test_list_experiments_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.list_experiments() == []


def test_create_experiment_generates_unique_ids() -> None:
    mgr, _ = _make_manager()
    exp1 = mgr.create_experiment(TrainingConfigData(model_name="a", experiment_name="a"))
    exp2 = mgr.create_experiment(TrainingConfigData(model_name="b", experiment_name="b"))
    assert exp1.experiment_id != exp2.experiment_id


def test_update_experiment_persists_best_metric() -> None:
    mgr, _ = _make_manager()
    cfg = TrainingConfigData(model_name="test", experiment_name="best_test")
    exp = mgr.create_experiment(cfg)
    updated = mgr.update_experiment(
        ExperimentData(
            experiment_id=exp.experiment_id,
            experiment_name=exp.experiment_name,
            config=exp.config,
            status="completed",
            best_epoch=5,
            best_metric=0.95,
            best_metric_name="mAP50",
        ),
    )
    assert updated.best_epoch == 5
    assert updated.best_metric == 0.95
