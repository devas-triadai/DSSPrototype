"""Tests for the EarlyStopping."""

from backend.training.early_stopping import EarlyStopping
from backend.training.models import EarlyStoppingConfig


def test_does_not_stop_early_when_improving() -> None:
    es = EarlyStopping(patience=3, min_delta=0.01)
    for epoch in range(10):
        should_stop = es.check(1.0 / (epoch + 1), epoch)
        if should_stop:
            break
    assert es.should_stop is False


def test_stops_when_no_improvement() -> None:
    es = EarlyStopping(patience=3, min_delta=0.01)
    es.check(0.5, 0)
    for epoch in range(1, 10):
        if es.check(0.5, epoch):
            break
    assert es.should_stop is True
    assert es.stopped_epoch <= 3


def test_stops_when_worsening() -> None:
    es = EarlyStopping(patience=2, min_delta=0.001, mode="max")
    es.check(0.5, 0)
    es.check(0.3, 1)
    es.check(0.3, 2)
    assert es.should_stop is True


def test_min_delta_respected() -> None:
    es = EarlyStopping(patience=2, min_delta=0.1)
    es.check(1.0, 0)
    # Within delta — not an improvement
    es.check(0.95, 1)
    assert es.counter == 1


def test_reset() -> None:
    es = EarlyStopping(patience=1, min_delta=0.01)
    es.check(0.5, 0)
    es.check(0.5, 1)
    assert es.should_stop is True
    es.reset()
    assert es.should_stop is False
    assert es.counter == 0
    assert es.best_value is None


def test_state_dict_roundtrip() -> None:
    es = EarlyStopping(patience=5)
    es.check(0.5, 0)
    es.check(0.6, 1)
    state = es.state_dict()
    es2 = EarlyStopping(patience=5)
    es2.load_state_dict(state)
    assert es2.counter == es.counter
    assert es2.best_value == es.best_value


def test_from_config() -> None:
    config = EarlyStoppingConfig(patience=10, min_delta=0.005, mode="max")
    es = EarlyStopping.from_config(config)
    assert es.patience == 10
    assert es.min_delta == 0.005
    assert es.mode == "max"
