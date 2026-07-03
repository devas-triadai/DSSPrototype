"""Tests for the CheckpointManager."""

import json
import tempfile
from pathlib import Path

from backend.training.checkpoint import CheckpointManager


def _make_manager() -> tuple[CheckpointManager, Path]:
    tmp = Path(tempfile.mkdtemp())
    return CheckpointManager(checkpoints_dir=tmp / "checkpoints"), tmp


def test_save_checkpoint() -> None:
    mgr, _ = _make_manager()
    ckpt = mgr.save_checkpoint("exp_001", epoch=1, metric_value=0.5)
    assert ckpt.experiment_id == "exp_001"
    assert ckpt.epoch == 1
    assert ckpt.metric_value == 0.5
    assert ckpt.is_latest is True


def test_best_checkpoint() -> None:
    mgr, _ = _make_manager()
    mgr.save_checkpoint("exp_002", epoch=1, metric_value=0.8)
    mgr.save_checkpoint("exp_002", epoch=2, metric_value=0.5)
    best = mgr.get_best_checkpoint("exp_002")
    assert best is not None
    assert best.epoch == 1
    assert best.metric_value == 0.8


def test_latest_checkpoint() -> None:
    mgr, _ = _make_manager()
    mgr.save_checkpoint("exp_003", epoch=1, metric_value=0.5)
    mgr.save_checkpoint("exp_003", epoch=2, metric_value=0.6)
    latest = mgr.get_latest_checkpoint("exp_003")
    assert latest is not None
    assert latest.epoch == 2
    assert latest.is_latest is True


def test_load_checkpoint() -> None:
    mgr, _ = _make_manager()
    mgr.save_checkpoint("exp_004", epoch=3, metric_value=0.9)
    ckpt = mgr.load_checkpoint("exp_004", epoch=3)
    assert ckpt is not None
    assert ckpt.epoch == 3


def test_load_checkpoint_not_found() -> None:
    mgr, _ = _make_manager()
    assert mgr.load_checkpoint("nonexistent", 1) is None


def test_list_checkpoints() -> None:
    mgr, _ = _make_manager()
    mgr.save_checkpoint("exp_005", epoch=1)
    mgr.save_checkpoint("exp_005", epoch=2)
    ckpts = mgr.list_checkpoints("exp_005")
    assert len(ckpts) == 2


def test_no_best_checkpoint_when_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_best_checkpoint("empty") is None


def test_no_latest_checkpoint_when_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_latest_checkpoint("empty") is None


def test_checkpoint_is_best_tracking() -> None:
    mgr, _ = _make_manager()
    ckpt1 = mgr.save_checkpoint("exp_best", epoch=1, metric_value=0.5)
    assert ckpt1.is_best is True
    ckpt2 = mgr.save_checkpoint("exp_best", epoch=2, metric_value=0.8)
    assert ckpt2.is_best is True
    ckpt1_reloaded = mgr.load_checkpoint("exp_best", epoch=1)
    assert ckpt1_reloaded is not None
    assert ckpt1_reloaded.is_best is False


def test_best_update_with_lower_value() -> None:
    mgr, _ = _make_manager()
    mgr.save_checkpoint("exp_lower", epoch=1, metric_value=0.9)
    mgr.save_checkpoint("exp_lower", epoch=2, metric_value=0.3)
    best = mgr.get_best_checkpoint("exp_lower")
    assert best is not None
    assert best.epoch == 1


def test_checkpoint_with_metadata() -> None:
    mgr, _ = _make_manager()
    meta: dict[str, object] = {"box_loss": 0.3, "cls_loss": 0.2}
    ckpt = mgr.save_checkpoint("exp_meta", epoch=1, metadata=meta)
    assert ckpt.metadata["box_loss"] == 0.3


def test_checkpoint_persists_to_disk() -> None:
    mgr, tmp = _make_manager()
    mgr.save_checkpoint("exp_disk", epoch=1, metric_value=0.5)
    meta_file = tmp / "checkpoints" / "exp_disk_epoch_0001.json"
    assert meta_file.exists()
    data = json.loads(meta_file.read_text())
    assert data["epoch"] == 1


def test_prune_old_checkpoints() -> None:
    mgr, _ = _make_manager()
    for e in range(1, 10):
        mgr.save_checkpoint("exp_prune", epoch=e, metric_value=float(e))
    ckpts = mgr.list_checkpoints("exp_prune")
    assert len(ckpts) <= 3


def test_checkpoints_dir_property() -> None:
    mgr, tmp = _make_manager()
    assert mgr.checkpoints_dir == tmp / "checkpoints"


def test_save_checkpoint_path_resolved() -> None:
    mgr, _ = _make_manager()
    ckpt = mgr.save_checkpoint("exp_path", epoch=1)
    assert "checkpoints" in ckpt.path
    assert "exp_path" in ckpt.path


def test_checkpoint_is_latest_flag() -> None:
    mgr, _ = _make_manager()
    mgr.save_checkpoint("exp_latest", epoch=1)
    mgr.save_checkpoint("exp_latest", epoch=2)
    ckpt1 = mgr.load_checkpoint("exp_latest", epoch=1)
    assert ckpt1 is not None
    assert ckpt1.is_latest is False
    ckpt2 = mgr.load_checkpoint("exp_latest", epoch=2)
    assert ckpt2 is not None
    assert ckpt2.is_latest is True


def test_checkpoint_list_empty() -> None:
    mgr, _ = _make_manager()
    assert mgr.list_checkpoints("empty") == []


def test_checkpoint_saved_at_set() -> None:
    mgr, _ = _make_manager()
    ckpt = mgr.save_checkpoint("exp_time", epoch=1)
    assert ckpt.saved_at != ""
    assert "T" in ckpt.saved_at
