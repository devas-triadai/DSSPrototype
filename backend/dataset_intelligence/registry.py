"""Dataset Intelligence Registry — persistent JSON-backed registry.

Tracks every dataset that has passed through the intelligence pipeline,
including its quality score, export paths, and report locations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from backend.dataset_intelligence.config import di_config
from backend.dataset_intelligence.exceptions import DatasetNotFoundError
from backend.dataset_intelligence.interfaces import DatasetIntelligenceRegistryInterface
from backend.dataset_intelligence.models import DatasetIntelligenceRegistryEntry

logger = logging.getLogger("dss.dataset_intelligence.registry")


class DatasetIntelligenceRegistry(DatasetIntelligenceRegistryInterface):
    """JSON-backed registry for processed datasets."""

    def __init__(self, registry_path: Path | None = None) -> None:
        self._path = registry_path or di_config.reports_dir / "dataset_intelligence_registry.json"
        self._entries: dict[str, DatasetIntelligenceRegistryEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw.get("entries", []):
                entry = DatasetIntelligenceRegistryEntry(**item)
                self._entries[entry.dataset_id] = entry
            logger.info("Registry loaded | entries=%d", len(self._entries))
        except Exception as exc:
            logger.warning("Failed to load registry: %s", exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            json.dump(
                {"entries": [e.model_dump() for e in self._entries.values()]},
                f,
                indent=2,
                default=str,
            )

    def register(self, entry: DatasetIntelligenceRegistryEntry) -> DatasetIntelligenceRegistryEntry:
        self._entries[entry.dataset_id] = entry
        self._save()
        logger.info("Registered dataset | id=%s | name=%s", entry.dataset_id, entry.dataset_name)
        return entry

    def get(self, dataset_id: str) -> DatasetIntelligenceRegistryEntry | None:
        return self._entries.get(dataset_id)

    def list_entries(self) -> list[DatasetIntelligenceRegistryEntry]:
        return sorted(self._entries.values(), key=lambda e: e.created_at)

    def update(self, entry: DatasetIntelligenceRegistryEntry) -> DatasetIntelligenceRegistryEntry:
        if entry.dataset_id not in self._entries:
            raise DatasetNotFoundError(f"Dataset not found: {entry.dataset_id}")
        self._entries[entry.dataset_id] = entry
        self._save()
        logger.info("Updated registry entry | id=%s", entry.dataset_id)
        return entry

    def delete(self, dataset_id: str) -> bool:
        if dataset_id not in self._entries:
            return False
        del self._entries[dataset_id]
        self._save()
        logger.info("Deleted registry entry | id=%s", dataset_id)
        return True
