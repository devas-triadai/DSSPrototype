"""Training backend registry — maps backend names to implementations.

Adding a new backend:
  1. Implement TrainingBackendInterface in a new module.
  2. Register it here::

        registry.register("detectron2", Detectron2Backend)

Future backends: Detectron2, RT-DETR, MMDetection, YOLO-NAS, Grounding DINO.
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.training.interfaces import TrainingBackendInterface

logger = logging.getLogger("dss.training.backends.registry")


class TrainingBackendRegistry:
    """Registry of available training backends.

    Backends are registered by name and instantiated on demand.
    """

    def __init__(self) -> None:
        self._backends: dict[str, type["TrainingBackendInterface"]] = {}

    def register(
        self, name: str, backend_cls: type["TrainingBackendInterface"],
    ) -> None:
        """Register a backend class under the given name."""
        self._backends[name] = backend_cls
        logger.info("Backend registered: %s → %s", name, backend_cls.__name__)

    def get(self, name: str) -> type["TrainingBackendInterface"]:
        """Get a backend class by name.

        Raises KeyError if the name is not registered.
        """
        if name not in self._backends:
            registered = list(self._backends.keys())
            raise KeyError(
                f"Unknown training backend: '{name}'. "
                f"Registered backends: {registered}",
            )
        return self._backends[name]

    def create(self, name: str, **kwargs: object) -> "TrainingBackendInterface":
        """Instantiate a backend by name with the given keyword arguments."""
        cls = self.get(name)
        return cls(**kwargs)

    def list_backends(self) -> list[str]:
        """Return the names of all registered backends."""
        return list(self._backends.keys())


_default_registry = TrainingBackendRegistry()


def get_default_registry() -> TrainingBackendRegistry:
    """Return the default global backend registry."""
    return _default_registry
