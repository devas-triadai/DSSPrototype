"""Model registry — tracks every trained model.

Provides a persistent registry of all models, their versions,
architectures, checkpoints, metrics, and export formats.
"""

import json
import logging
import uuid
from pathlib import Path

from backend.training.config import training_config
from backend.training.exceptions import ModelNotFoundError
from backend.training.interfaces import ModelRegistryInterface
from backend.training.models import ModelEntry

logger = logging.getLogger("dss.training.registry")


class ModelRegistry(ModelRegistryInterface):
    """Persistent model registry using JSON storage.

    Each model is stored as a JSON file in the models directory.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        self._config = training_config
        self._models_dir = models_dir or self._config.models_dir
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, ModelEntry] = {}

    def register_model(self, entry: ModelEntry) -> ModelEntry:
        logger.info("Model registration started: %s (%s)", entry.model_name, entry.model_id)
        self._cache[entry.model_id] = entry
        self._persist(entry)
        logger.info("Model registered: %s v%s", entry.model_name, entry.version)
        return entry

    def get_model(self, model_id: str) -> ModelEntry | None:
        if model_id in self._cache:
            return self._cache[model_id]
        path = self._model_path(model_id)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            entry = ModelEntry(**data)
            self._cache[model_id] = entry
            return entry
        except Exception:
            return None

    def get_models_by_name(self, name: str) -> list[ModelEntry]:
        return [m for m in self.list_models() if m.model_name == name]

    def list_models(self) -> list[ModelEntry]:
        models: list[ModelEntry] = []
        for path in sorted(self._models_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                models.append(ModelEntry(**data))
            except Exception:
                pass
        return models

    def update_model(self, entry: ModelEntry) -> ModelEntry:
        if not self._model_path(entry.model_id).exists() and entry.model_id not in self._cache:
            raise ModelNotFoundError(f"Model not found: {entry.model_id}")

        updated = ModelEntry(
            model_id=entry.model_id,
            model_name=entry.model_name,
            architecture=entry.architecture,
            version=entry.version,
            dataset_version=entry.dataset_version,
            experiment_id=entry.experiment_id,
            checkpoint_path=entry.checkpoint_path,
            metrics=entry.metrics,
            training_status=entry.training_status,
            framework=entry.framework,
            export_formats=entry.export_formats,
            checksum=entry.checksum,
        )
        self._cache[entry.model_id] = updated
        self._persist(updated)
        logger.info("Model updated: %s (status=%s)", entry.model_id, entry.training_status)
        return updated

    def delete_model(self, model_id: str) -> bool:
        self._cache.pop(model_id, None)
        path = self._model_path(model_id)
        if not path.exists():
            return False
        path.unlink()
        logger.info("Model deleted: %s", model_id)
        return True

    def _generate_id(self) -> str:
        return f"model_{uuid.uuid4().hex[:12]}"

    def _model_path(self, model_id: str) -> Path:
        return self._models_dir / f"{model_id}.json"

    def _persist(self, entry: ModelEntry) -> None:
        path = self._model_path(entry.model_id)
        path.write_text(json.dumps(entry.model_dump(), indent=2, default=str))
