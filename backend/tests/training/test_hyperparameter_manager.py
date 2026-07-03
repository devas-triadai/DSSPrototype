"""Tests for the HyperparameterManager."""

import tempfile
from pathlib import Path

from backend.training.hyperparameter_manager import HyperparameterManager
from backend.training.models import HyperparameterProfile


def _make_manager() -> tuple[HyperparameterManager, Path]:
    tmp = Path(tempfile.mkdtemp())
    return HyperparameterManager(configs_dir=tmp / "configs"), tmp


def test_builtin_profiles_loaded() -> None:
    mgr, _ = _make_manager()
    profiles = mgr.list_profiles()
    names = {p.name for p in profiles}
    assert "fast" in names
    assert "balanced" in names
    assert "accurate" in names
    assert "tiny" in names


def test_get_fast_profile() -> None:
    mgr, _ = _make_manager()
    p = mgr.get_profile("fast")
    assert p is not None
    assert p.name == "fast"
    assert p.batch_size == 32
    assert p.epochs == 10
    assert p.image_size == (416, 416)


def test_get_balanced_profile() -> None:
    mgr, _ = _make_manager()
    p = mgr.get_profile("balanced")
    assert p is not None
    assert p.batch_size == 16
    assert p.epochs == 100


def test_get_accurate_profile() -> None:
    mgr, _ = _make_manager()
    p = mgr.get_profile("accurate")
    assert p is not None
    assert p.batch_size == 8
    assert p.epochs == 300
    assert p.mixed_precision is True
    assert p.warmup_epochs == 3


def test_get_tiny_profile() -> None:
    mgr, _ = _make_manager()
    p = mgr.get_profile("tiny")
    assert p is not None
    assert p.batch_size == 64
    assert p.epochs == 50
    assert p.optimizer == "sgd"
    assert p.image_size == (320, 320)


def test_get_profile_not_found() -> None:
    mgr, _ = _make_manager()
    assert mgr.get_profile("nonexistent") is None


def test_save_and_get_profile() -> None:
    mgr, _ = _make_manager()
    p = HyperparameterProfile(name="custom", learning_rate=0.01, batch_size=8, epochs=20)
    saved = mgr.save_profile(p)
    assert saved.name == "custom"
    assert mgr.get_profile("custom") is not None


def test_save_profile_persists_to_disk() -> None:
    mgr, tmp = _make_manager()
    p = HyperparameterProfile(name="disk_test", learning_rate=0.01, batch_size=8, epochs=20)
    mgr.save_profile(p)
    configs = list((tmp / "configs").glob("*.json"))
    assert len(configs) >= 1


def test_list_profiles_includes_saved() -> None:
    mgr, _ = _make_manager()
    mgr.save_profile(HyperparameterProfile(name="custom1", batch_size=4, epochs=5))
    mgr.save_profile(HyperparameterProfile(name="custom2", batch_size=2, epochs=3))
    names = {p.name for p in mgr.list_profiles()}
    assert "custom1" in names
    assert "custom2" in names


def test_delete_custom_profile() -> None:
    mgr, _ = _make_manager()
    mgr.save_profile(HyperparameterProfile(name="delete_me", batch_size=4, epochs=5))
    assert mgr.delete_profile("delete_me") is True
    assert mgr.get_profile("delete_me") is None


def test_delete_profile_not_found() -> None:
    mgr, _ = _make_manager()
    assert mgr.delete_profile("nonexistent") is False


def test_cannot_delete_builtin_fast() -> None:
    mgr, _ = _make_manager()
    assert mgr.delete_profile("fast") is False
    assert mgr.get_profile("fast") is not None


def test_cannot_delete_builtin_balanced() -> None:
    mgr, _ = _make_manager()
    assert mgr.delete_profile("balanced") is False


def test_cannot_delete_builtin_accurate() -> None:
    mgr, _ = _make_manager()
    assert mgr.delete_profile("accurate") is False


def test_cannot_delete_builtin_tiny() -> None:
    mgr, _ = _make_manager()
    assert mgr.delete_profile("tiny") is False


def test_apply_profile() -> None:
    mgr, _ = _make_manager()
    profile = mgr.get_profile("fast")
    assert profile is not None
    config = mgr.apply_profile(profile)
    assert config.batch_size == 32
    assert config.epochs == 10
    assert config.learning_rate == 0.001


def test_apply_profile_with_overrides() -> None:
    mgr, _ = _make_manager()
    profile = mgr.get_profile("fast")
    assert profile is not None
    config = mgr.apply_profile(profile, overrides={"batch_size": 64, "epochs": 5})
    assert config.batch_size == 64
    assert config.epochs == 5
    assert config.learning_rate == 0.001


def test_apply_profile_with_image_size_override() -> None:
    mgr, _ = _make_manager()
    profile = mgr.get_profile("fast")
    assert profile is not None
    config = mgr.apply_profile(profile, overrides={"image_size": (800, 800)})
    assert config.image_size == (800, 800)


def test_apply_profile_with_early_stopping() -> None:
    mgr, _ = _make_manager()
    profile = HyperparameterProfile(
        name="with_es", learning_rate=0.001, batch_size=16,
        epochs=100, early_stopping_patience=10,
    )
    config = mgr.apply_profile(profile)
    assert config.early_stopping_patience == 10


def test_list_profiles_no_duplicates() -> None:
    mgr, _ = _make_manager()
    profiles = mgr.list_profiles()
    names = [p.name for p in profiles]
    assert len(names) == len(set(names))
