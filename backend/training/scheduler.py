"""Learning rate scheduler abstraction.

Provides a model-agnostic scheduler that supports cosine, step,
linear, and polynomial decay schedules. New schedules are added
by extending the ``_SCHEDULERS`` registry.
"""

import logging
import math

from backend.training.interfaces import SchedulerInterface

logger = logging.getLogger("dss.training.scheduler")


class Scheduler(SchedulerInterface):
    """Model-agnostic learning rate scheduler.

    Supports: cosine, step, linear, polynomial.

    Example::

        scheduler = Scheduler(name="cosine", warmup_epochs=5)
        lr = scheduler.get_lr(epoch=10, base_lr=0.001)
    """

    def __init__(
        self,
        name: str = "cosine",
        warmup_epochs: int = 0,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 1e-6,
        **params: float,
    ) -> None:
        self._name = name
        self._warmup_epochs = warmup_epochs
        self._warmup_start_lr = warmup_start_lr
        self._min_lr = min_lr
        self._params = params
        self._state: dict[str, object] = {}

    def get_lr(self, epoch: int, base_lr: float) -> float:
        if epoch < self._warmup_epochs:
            return self._warmup_lr(epoch, base_lr)

        adjusted_epoch = epoch - self._warmup_epochs
        total_epochs = int(self._params.get("total_epochs", 100))

        if self._name == "cosine":
            return self._cosine(adjusted_epoch, base_lr, total_epochs)
        elif self._name == "step":
            return self._step(adjusted_epoch, base_lr)
        elif self._name == "linear":
            return self._linear(adjusted_epoch, base_lr, total_epochs)
        elif self._name == "polynomial":
            return self._polynomial(adjusted_epoch, base_lr, total_epochs)
        else:
            return base_lr

    def state_dict(self) -> dict[str, object]:
        return {
            "name": self._name,
            "warmup_epochs": self._warmup_epochs,
            "warmup_start_lr": self._warmup_start_lr,
            "min_lr": self._min_lr,
            "params": self._params,
            "state": self._state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self._name = str(state.get("name", "cosine"))
        self._warmup_epochs = int(state.get("warmup_epochs", 0))  # type: ignore[call-overload]
        self._warmup_start_lr = float(state.get("warmup_start_lr", 1e-6))  # type: ignore[arg-type]
        self._min_lr = float(state.get("min_lr", 1e-6))  # type: ignore[arg-type]
        self._params = dict(state.get("params", {}))  # type: ignore[call-overload]
        self._state = dict(state.get("state", {}))  # type: ignore[call-overload]

    def _warmup_lr(self, epoch: int, base_lr: float) -> float:
        progress = epoch / max(self._warmup_epochs, 1)
        return self._warmup_start_lr + (base_lr - self._warmup_start_lr) * progress

    def _cosine(self, epoch: int, base_lr: float, total_epochs: int) -> float:
        progress = epoch / max(total_epochs, 1)
        return self._min_lr + 0.5 * (base_lr - self._min_lr) * (1.0 + math.cos(math.pi * progress))

    def _step(self, epoch: int, base_lr: float) -> float:
        step_size = int(self._params.get("step_size", 30))
        gamma = float(self._params.get("gamma", 0.1))
        return base_lr * (gamma ** (epoch // step_size))

    def _linear(self, epoch: int, base_lr: float, total_epochs: int) -> float:
        progress = epoch / max(total_epochs, 1)
        return base_lr * (1.0 - progress) + self._min_lr * progress

    def _polynomial(self, epoch: int, base_lr: float, total_epochs: int) -> float:
        power_val: object = self._params.get("power", 2.0)
        power = float(power_val) if isinstance(power_val, (int, float)) else 2.0
        progress = epoch / max(total_epochs, 1)
        result_val = (base_lr - self._min_lr) * ((1.0 - progress) ** power) + self._min_lr
        return result_val
