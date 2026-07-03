"""Model-agnostic model lifecycle manager with a plugin registry.

New vision model types register themselves via ``register_model()``
so the manager can instantiate them without knowing their concrete type.
"""

import logging

from backend.modules.computer_vision.config import cv_config
from backend.modules.computer_vision.exceptions import ModelLoadError
from backend.modules.computer_vision.interfaces import (
    ModelManagerInterface,
    ModelMetadata,
    VisionModelInterface,
)

logger = logging.getLogger("dss.computer_vision.model_manager")

# ---------------------------------------------------------------------------
# Plugin registry — populated by model implementations at import time
# ---------------------------------------------------------------------------

_registry: dict[str, type[VisionModelInterface]] = {}


def register_model(model_type: str, model_class: type[VisionModelInterface]) -> None:
    """Register a model class so ``ModelManager`` can instantiate it by type.

    Called once at module import time by each model implementation::

        from backend.modules.computer_vision.model_manager import register_model
        from backend.modules.computer_vision.interfaces import VisionModelInterface

        class YOLOModel(VisionModelInterface):
            ...

        register_model("yolo", YOLOModel)
    """
    if model_type in _registry:
        logger.warning("Overwriting existing registration for model type '%s'", model_type)
    _registry[model_type] = model_class
    logger.info("Registered model type '%s' -> %s", model_type, model_class.__name__)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class ModelManager(ModelManagerInterface):
    """Manages model lifecycle and provides access to loaded models.

    Usage::

        manager = ModelManager()
        model = manager.load_model("yolo", "/path/to/model.pt")
        result = model.predict(image)
    """

    def __init__(self) -> None:
        self._config = cv_config
        self._models: dict[str, VisionModelInterface] = {}

    def load_model(self, model_type: str, model_path: str) -> VisionModelInterface:
        """Load a model of *model_type* from *model_path*.

        Parameters
        ----------
        model_type:
            Registered model type key (e.g. ``"yolo"``, ``"rt-detr"``).
        model_path:
            Filesystem path to the model weights.

        Returns
        -------
        VisionModelInterface
            The loaded model instance.
        """
        if model_type in self._models:
            logger.info("Model '%s' already loaded, returning cached instance", model_type)
            return self._models[model_type]

        model_class = _registry.get(model_type)
        if model_class is None:
            registered = list(_registry)
            raise ModelLoadError(
                f"Unknown model type '{model_type}'. "
                f"Registered types: {registered}. "
                f"Did you forget to import the model plugin?"
            )

        model = model_class()
        try:
            model.load()
        except Exception as exc:
            raise ModelLoadError(f"Failed to load model '{model_type}': {exc}") from exc

        self._models[model_type] = model
        logger.info("Loaded model '%s' from %s", model_type, model_path)
        return model

    def get_model(self, model_type: str) -> VisionModelInterface:
        """Retrieve a previously loaded model.

        Raises
        ------
        ModelLoadError
            If the model has not been loaded yet.
        """
        if model_type not in self._models:
            raise ModelLoadError(f"Model '{model_type}' is not loaded")
        return self._models[model_type]

    def unload_model(self, model_type: str) -> None:
        """Unload a model and free its resources."""
        model = self._models.pop(model_type, None)
        if model is not None:
            model.unload()
            logger.info("Unloaded model '%s'", model_type)

    def reload_model(self, model_type: str) -> None:
        """Reload a model with the current configuration."""
        self.unload_model(model_type)
        self.load_model(model_type, self._config.model_path)

    def list_models(self) -> list[ModelMetadata]:
        """Return metadata for all loaded models."""
        return [m.metadata for m in self._models.values()]
