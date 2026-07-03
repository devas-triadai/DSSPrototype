"""Tests for the HistoryManager."""

import json
import tempfile
from pathlib import Path

from backend.training.history import HistoryManager
from backend.training.models import HistoryEntry


def _make_manager() -> tuple[HistoryManager, Path]:
    tmp = Path(tempfile.mkdtemp())
    return HistoryManager(history_dir=tmp / "history"), tmp


def test_record_and_get() -> None:
    mgr, _ = _make_manager()
    entry = HistoryEntry(experiment_id="exp_001", epoch=1, loss=0.5, learning_rate=0.001)
    mgr.record_entry(entry)
    history = mgr.get_history("exp_001")
    assert len(history) == 1
    assert history[0].loss == 0.5


def test_get_latest_entry() -> None:
    mgr, _ = _make_manager()
    mgr.record_entry(HistoryEntry(experiment_id="exp_002", epoch=1, loss=0.8, learning_rate=0.001))
    mgr.record_entry(HistoryEntry(experiment_id="exp_002", epoch=2, loss=0.5, learning_rate=0.0001))
    latest = mgr.get_latest_entry("exp_002")
    assert latest is not None
    assert latest.loss == 0.5
    assert latest.epoch == 2


def test_get_history_as_dicts() -> None:
    mgr, _ = _make_manager()
    mgr.record_entry(HistoryEntry(experiment_id="exp_003", epoch=1, loss=0.5, learning_rate=0.001))
    dicts = mgr.get_history_as_dicts("exp_003")
    assert len(dicts) == 1
    assert dicts[0]["epoch"] == 1
    assert dicts[0]["loss"] == 0.5


def test_empty_history() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_history("empty") == []
    assert mgr.get_latest_entry("empty") is None


def test_multiple_experiments_separate() -> None:
    mgr, _ = _make_manager()
    mgr.record_entry(HistoryEntry(experiment_id="exp_a", epoch=1, loss=0.5, learning_rate=0.001))
    mgr.record_entry(HistoryEntry(experiment_id="exp_b", epoch=1, loss=0.3, learning_rate=0.001))
    assert len(mgr.get_history("exp_a")) == 1
    assert len(mgr.get_history("exp_b")) == 1


def test_record_entry_persists_to_disk() -> None:
    mgr, tmp = _make_manager()
    entry = HistoryEntry(experiment_id="exp_disk", epoch=1, loss=0.5, learning_rate=0.001)
    mgr.record_entry(entry)
    hist_file = tmp / "history" / "exp_disk_history.json"
    assert hist_file.exists()
    data = json.loads(hist_file.read_text())
    assert len(data) == 1
    assert data[0]["epoch"] == 1


def test_get_latest_entry_empty_after_delete() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_latest_entry("no_data") is None


def test_history_with_metrics_dict() -> None:
    mgr, _ = _make_manager()
    entry = HistoryEntry(
        experiment_id="exp_met", epoch=1, loss=0.5, learning_rate=0.001,
        metrics={"mAP50": 0.85, "precision": 0.9},
    )
    mgr.record_entry(entry)
    history = mgr.get_history("exp_met")
    assert history[0].metrics["mAP50"] == 0.85


def test_history_order_preserved() -> None:
    mgr, _ = _make_manager()
    mgr.record_entry(HistoryEntry(experiment_id="exp_order", epoch=1, loss=0.9, learning_rate=0.001))  # noqa: E501
    mgr.record_entry(HistoryEntry(experiment_id="exp_order", epoch=2, loss=0.5, learning_rate=0.001))  # noqa: E501
    mgr.record_entry(HistoryEntry(experiment_id="exp_order", epoch=3, loss=0.3, learning_rate=0.001))  # noqa: E501
    history = mgr.get_history("exp_order")
    assert [e.epoch for e in history] == [1, 2, 3]
