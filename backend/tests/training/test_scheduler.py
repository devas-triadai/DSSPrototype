"""Tests for the Scheduler."""

import pytest

from backend.training.scheduler import Scheduler


def test_cosine_scheduler() -> None:
    s = Scheduler(name="cosine", total_epochs=100)
    lr_0 = s.get_lr(0, 0.001)
    lr_50 = s.get_lr(50, 0.001)
    lr_100 = s.get_lr(100, 0.001)
    assert lr_0 == 0.001
    assert lr_50 < lr_0
    assert lr_100 < lr_50


def test_step_scheduler() -> None:
    s = Scheduler(name="step", step_size=30, gamma=0.1, total_epochs=100)
    lr_0 = s.get_lr(0, 0.001)
    lr_30 = s.get_lr(30, 0.001)
    lr_60 = s.get_lr(60, 0.001)
    assert lr_0 == 0.001
    assert lr_30 == 0.001 * 0.1
    assert lr_60 == pytest.approx(0.001 * 0.01, rel=1e-12)


def test_linear_scheduler() -> None:
    s = Scheduler(name="linear", total_epochs=100)
    lr_0 = s.get_lr(0, 0.001)
    lr_50 = s.get_lr(50, 0.001)
    lr_100 = s.get_lr(100, 0.001)
    assert lr_0 == 0.001
    assert lr_50 < lr_0
    assert lr_100 < lr_50


def test_polynomial_scheduler() -> None:
    s = Scheduler(name="polynomial", power=2.0, total_epochs=100)
    lr_0 = s.get_lr(0, 0.001)
    lr_100 = s.get_lr(100, 0.001)
    assert lr_0 == 0.001
    assert lr_100 < 0.001
    assert s.state_dict() is not None


def test_warmup() -> None:
    s = Scheduler(name="cosine", warmup_epochs=5, warmup_start_lr=1e-6, total_epochs=100)
    lr_0 = s.get_lr(0, 0.001)
    lr_4 = s.get_lr(4, 0.001)
    lr_5 = s.get_lr(5, 0.001)
    assert lr_0 > 0
    assert lr_0 < lr_4
    assert lr_4 < lr_5


def test_state_dict_roundtrip() -> None:
    s = Scheduler(name="cosine", total_epochs=100)
    state = s.state_dict()
    s2 = Scheduler()
    s2.load_state_dict(state)
    assert s.get_lr(10, 0.001) == s2.get_lr(10, 0.001)


def test_unknown_scheduler_falls_back() -> None:
    s = Scheduler(name="unknown", total_epochs=100)
    assert s.get_lr(5, 0.001) == 0.001
