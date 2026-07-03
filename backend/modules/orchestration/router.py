"""Module registry that maps type keys to concrete module instances.

All lookups go through ``RouterInterface`` so the orchestration
engine depends on abstractions, not concrete implementations.
"""

from typing import Any

from backend.modules.orchestration.exceptions import RoutingError
from backend.modules.orchestration.interfaces import RouterInterface


class Router(RouterInterface):
    """Registry of module instances keyed by type string.

    Usage::

        router = Router()
        router.register("computer_vision", vision_service)
        router.register("friendly", friendly_service)

        vision = router.get("computer_vision")  # → VisionModule
    """

    def __init__(self) -> None:
        self._modules: dict[str, Any] = {}

    def register(self, module_type: str, module: Any) -> None:
        """Register a module under *module_type*.

        Parameters
        ----------
        module_type:
            Unique key for the module (e.g. ``"computer_vision"``).
        module:
            The module instance to register.

        Raises
        ------
        RoutingError
            If *module_type* is already registered.
        """
        if module_type in self._modules:
            raise RoutingError(f"Module '{module_type}' is already registered")
        self._modules[module_type] = module

    def get(self, module_type: str) -> Any:
        """Retrieve a module by type key.

        Raises
        ------
        RoutingError
            If the type is not registered.
        """
        module = self._modules.get(module_type)
        if module is None:
            registered = list(self._modules)
            raise RoutingError(
                f"No module registered for '{module_type}'. "
                f"Registered types: {registered}"
            )
        return module

    def registered_types(self) -> list[str]:
        """Return all registered type keys."""
        return list(self._modules)
