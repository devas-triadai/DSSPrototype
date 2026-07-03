"""Callback system — injectable hooks into the training loop.

Supports:
  - on_train_start
  - on_epoch_start
  - on_epoch_end
  - on_validation_end
  - on_checkpoint_saved
  - on_train_end

Custom callbacks implement the Callback ABC and are injected
into the Trainer at construction time.
"""

import logging
from abc import ABC
from collections.abc import Sequence

from backend.training.models import CheckpointData, ExportData, HistoryEntry, MetricData

logger = logging.getLogger("dss.training.callbacks")


class Callback(ABC):
    """Abstract base class for all training callbacks.

    Subclass and override the hooks you need.
    """

    def on_train_start(self) -> None:
        """Called once when training begins."""

    def on_epoch_start(self, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(
        self, epoch: int, history_entry: HistoryEntry | None = None,
    ) -> None:
        """Called at the end of each epoch."""

    def on_validation_end(self, metrics: MetricData | None = None) -> None:
        """Called after validation completes."""

    def on_checkpoint_saved(self, checkpoint: CheckpointData | None = None) -> None:
        """Called after a checkpoint is saved."""

    def on_export(self, export_data: "ExportData | None" = None) -> None:
        """Called after a model is exported."""

    def on_train_end(self) -> None:
        """Called once when training ends (success or interrupt)."""


class CallbackRunner:
    """Runs a sequence of callbacks, tolerating individual failures."""

    def __init__(self, callbacks: Sequence[Callback] | None = None) -> None:
        self._callbacks: list[Callback] = list(callbacks) if callbacks else []

    def add_callback(self, callback: Callback) -> None:
        self._callbacks.append(callback)

    def on_train_start(self) -> None:
        for cb in self._callbacks:
            try:
                cb.on_train_start()
            except Exception as e:
                logger.warning("Callback %s.on_train_start failed: %s", type(cb).__name__, e)

    def on_epoch_start(self, epoch: int) -> None:
        for cb in self._callbacks:
            try:
                cb.on_epoch_start(epoch)
            except Exception as e:
                logger.warning("Callback %s.on_epoch_start failed: %s", type(cb).__name__, e)

    def on_epoch_end(self, epoch: int, history_entry: HistoryEntry | None = None) -> None:
        for cb in self._callbacks:
            try:
                cb.on_epoch_end(epoch, history_entry)
            except Exception as e:
                logger.warning("Callback %s.on_epoch_end failed: %s", type(cb).__name__, e)

    def on_validation_end(self, metrics: MetricData | None = None) -> None:
        for cb in self._callbacks:
            try:
                cb.on_validation_end(metrics)
            except Exception as e:
                logger.warning("Callback %s.on_validation_end failed: %s", type(cb).__name__, e)

    def on_checkpoint_saved(self, checkpoint: CheckpointData | None = None) -> None:
        for cb in self._callbacks:
            try:
                cb.on_checkpoint_saved(checkpoint)
            except Exception as e:
                logger.warning("Callback %s.on_checkpoint_saved failed: %s", type(cb).__name__, e)

    def on_export(self, export_data: ExportData | None = None) -> None:
        for cb in self._callbacks:
            try:
                cb.on_export(export_data)
            except Exception as e:
                logger.warning("Callback %s.on_export failed: %s", type(cb).__name__, e)

    def on_train_end(self) -> None:
        for cb in self._callbacks:
            try:
                cb.on_train_end()
            except Exception as e:
                logger.warning("Callback %s.on_train_end failed: %s", type(cb).__name__, e)
