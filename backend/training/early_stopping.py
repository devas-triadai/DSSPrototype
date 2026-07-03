"""Early stopping — reusable, model-agnostic early stopping component.

Monitors a metric during training and signals when to stop based on
patience and minimum delta. Optionally tracks the best checkpoint.
"""

import logging

from backend.training.models import EarlyStoppingConfig

logger = logging.getLogger("dss.training.early_stopping")


class EarlyStopping:
    """Model-agnostic early stopping.

    Example::

        early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        for epoch in range(epochs):
            val_loss = validate(model)
            if early_stopping.check(val_loss, epoch):
                print("Early stopping triggered at epoch", epoch)
                break
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        monitor: str = "validation_loss",
        mode: str = "min",
        restore_best_checkpoint: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_checkpoint = restore_best_checkpoint

        self.best_value: float | None = None
        self.best_epoch: int | None = None
        self.counter: int = 0
        self.stopped_epoch: int = 0
        self.should_stop: bool = False

        logger.info(
            "EarlyStopping initialized: patience=%d, delta=%.4f, mode=%s",
            patience, min_delta, mode,
        )

    def check(self, metric_value: float, epoch: int) -> bool:
        """Check whether training should stop.

        Call this after each validation epoch.

        Parameters
        ----------
        metric_value:
            The current epoch's metric value.
        epoch:
            The current epoch number.

        Returns
        -------
        bool
            True if training should stop.
        """
        if self.best_value is None:
            self.best_value = metric_value
            self.best_epoch = epoch
            return False

        improved = self._is_improvement(metric_value, self.best_value)

        if improved:
            self.best_value = metric_value
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                "EarlyStopping counter: %d / %d (best=%.4f, current=%.4f)",
                self.counter, self.patience, self.best_value, metric_value,
            )

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            logger.info(
                "Early stopping triggered at epoch %d (best=%.4f)",
                epoch, self.best_value,
            )
            return True

        return False

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta

    def reset(self) -> None:
        """Reset early stopping state for a new training run."""
        self.best_value = None
        self.best_epoch = None
        self.counter = 0
        self.stopped_epoch = 0
        self.should_stop = False

    def state_dict(self) -> dict[str, object]:
        return {
            "best_value": self.best_value,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "stopped_epoch": self.stopped_epoch,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.best_value = state.get("best_value")  # type: ignore[assignment]
        self.best_epoch = state.get("best_epoch")  # type: ignore[assignment]
        self.counter = int(state.get("counter", 0))  # type: ignore[call-overload]
        self.stopped_epoch = int(state.get("stopped_epoch", 0))  # type: ignore[call-overload]
        self.should_stop = bool(state.get("should_stop", False))

    @classmethod
    def from_config(cls, config: EarlyStoppingConfig) -> "EarlyStopping":
        return cls(
            patience=config.patience,
            min_delta=config.min_delta,
            monitor=config.monitor,
            mode=config.mode,
            restore_best_checkpoint=config.restore_best_checkpoint,
        )
