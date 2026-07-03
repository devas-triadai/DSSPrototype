"""Tests for the Callback system."""

from backend.training.callbacks import Callback, CallbackRunner
from backend.training.models import CheckpointData, HistoryEntry, MetricData


class _TestCallback(Callback):
    def __init__(self) -> None:
        self.events: list[str] = []

    def on_train_start(self) -> None:
        self.events.append("train_start")

    def on_epoch_start(self, epoch: int) -> None:
        self.events.append(f"epoch_start:{epoch}")

    def on_epoch_end(self, epoch: int, history_entry: HistoryEntry | None = None) -> None:
        self.events.append(f"epoch_end:{epoch}")

    def on_validation_end(self, metrics: MetricData | None = None) -> None:
        self.events.append("validation_end")

    def on_checkpoint_saved(self, checkpoint: CheckpointData | None = None) -> None:
        self.events.append("checkpoint_saved")

    def on_train_end(self) -> None:
        self.events.append("train_end")


class _FailingCallback(Callback):
    def on_train_start(self) -> None:
        raise RuntimeError("callback failed")


def test_callback_runner_events() -> None:
    cb = _TestCallback()
    runner = CallbackRunner([cb])
    runner.on_train_start()
    runner.on_epoch_start(1)
    runner.on_epoch_end(1, None)
    runner.on_validation_end(None)
    runner.on_checkpoint_saved(None)
    runner.on_train_end()
    assert cb.events == [
        "train_start",
        "epoch_start:1",
        "epoch_end:1",
        "validation_end",
        "checkpoint_saved",
        "train_end",
    ]


def test_callback_runner_empty() -> None:
    runner = CallbackRunner()
    runner.on_train_start()
    assert True


def test_callback_runner_exception_isolated() -> None:
    good = _TestCallback()
    bad = _FailingCallback()
    runner = CallbackRunner([good, bad])
    runner.on_train_start()
    assert "train_start" in good.events


def test_add_callback() -> None:
    cb = _TestCallback()
    runner = CallbackRunner()
    runner.add_callback(cb)
    runner.on_train_start()
    assert cb.events == ["train_start"]
