"""Module registry for the Runtime Integration Layer.

Registers runtime modules using their contract interfaces only.
Supports dependency injection with no concrete dependencies.
"""

import logging
from typing import Any, cast

from backend.contracts.interfaces.decision import DecisionModule
from backend.contracts.interfaces.enemy import EnemyModule
from backend.contracts.interfaces.friendly import FriendlyModule
from backend.contracts.interfaces.fusion import FusionModule
from backend.contracts.interfaces.terrain import TerrainModule
from backend.contracts.interfaces.vision import VisionModule

logger = logging.getLogger("dss.orchestration.module_registry")

MODULE_TYPE_CV = "computer_vision"
MODULE_TYPE_FRIENDLY = "friendly"
MODULE_TYPE_ENEMY = "enemy"
MODULE_TYPE_TERRAIN = "terrain"
MODULE_TYPE_FUSION = "fusion"
MODULE_TYPE_DECISION = "decision"

_REQUIRED_MODULES: set[str] = {
    MODULE_TYPE_CV,
    MODULE_TYPE_FRIENDLY,
    MODULE_TYPE_ENEMY,
    MODULE_TYPE_TERRAIN,
    MODULE_TYPE_FUSION,
    MODULE_TYPE_DECISION,
}


class ModuleRegistry:
    """Registers runtime modules using their contract interfaces.

    All registration uses the abstract interface types.
    No concrete implementations are referenced in this class.
    """

    def __init__(self) -> None:
        self._modules: dict[str, Any] = {}

    def register(self, module_type: str, module: Any) -> None:
        """Register a module instance under *module_type*.

        Parameters
        ----------
        module_type:
            Unique key identifying the module (e.g. ``"computer_vision"``).
        module:
            The module instance (must implement the corresponding interface).
        """
        if module_type in self._modules:
            logger.warning("Overwriting existing module '%s'", module_type)
        self._modules[module_type] = module
        logger.debug("Registered module '%s': %s", module_type, type(module).__name__)

    def get(self, module_type: str) -> Any:
        """Retrieve a registered module by type key.

        Raises
        ------
        RuntimeError
            If the module type has not been registered.
        """
        module = self._modules.get(module_type)
        if module is None:
            registered = list(self._modules)
            raise RuntimeError(
                f"No module registered for '{module_type}'. "
                f"Registered types: {registered}"
            )
        return module

    # ------------------------------------------------------------------
    # Typed accessors
    # ------------------------------------------------------------------

    def get_vision(self) -> VisionModule:
        """Return the registered Computer Vision module."""
        return cast(VisionModule, self.get(MODULE_TYPE_CV))

    def get_friendly(self) -> FriendlyModule:
        """Return the registered Friendly Knowledge module."""
        return cast(FriendlyModule, self.get(MODULE_TYPE_FRIENDLY))

    def get_enemy(self) -> EnemyModule:
        """Return the registered Enemy Knowledge module."""
        return cast(EnemyModule, self.get(MODULE_TYPE_ENEMY))

    def get_terrain(self) -> TerrainModule:
        """Return the registered Terrain Knowledge module."""
        return cast(TerrainModule, self.get(MODULE_TYPE_TERRAIN))

    def get_fusion(self) -> FusionModule:
        """Return the registered Fusion Engine module."""
        return cast(FusionModule, self.get(MODULE_TYPE_FUSION))

    def get_decision(self) -> DecisionModule:
        """Return the registered Decision Engine module."""
        return cast(DecisionModule, self.get(MODULE_TYPE_DECISION))

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def registered_types(self) -> list[str]:
        """Return all registered module type keys."""
        return list(self._modules)

    def is_complete(self) -> bool:
        """Return True if all required modules are registered."""
        return _REQUIRED_MODULES.issubset(self._modules.keys())

    def missing_modules(self) -> list[str]:
        """Return sorted list of required modules not yet registered."""
        return sorted(_REQUIRED_MODULES - self._modules.keys())
