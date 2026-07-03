"""JSON-backed persistent catalog store."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from threading import Lock

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import CatalogError, EntryNotFoundError
from backend.dataset_catalog.interfaces import CatalogInterface
from backend.dataset_catalog.models import CatalogEntry

logger = logging.getLogger("dss.dataset_catalog.catalog")


class Catalog(CatalogInterface):
    """JSON-file-backed persistent catalog store.

    Thread-safe via Lock. Persists catalog entries to a single JSON file.
    The store is loaded on construction and written after every mutation.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or dc_config.catalog_db_path
        self._lock = Lock()
        self._entries: dict[str, CatalogEntry] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        if self._db_path.exists():
            try:
                with self._db_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._entries = {
                    eid: CatalogEntry(**data) for eid, data in raw.items()
                }
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load catalog DB: %s — starting fresh", exc)
                self._entries = {}
        else:
            self._entries = {}

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            eid: entry.model_dump(mode="json") for eid, entry in self._entries.items()
        }
        with self._db_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_entry(self, entry: CatalogEntry) -> CatalogEntry:
        with self._lock:
            if entry.entry_id in self._entries:
                raise CatalogError(f"Entry already exists: {entry.entry_id}")
            self._entries[entry.entry_id] = entry
            self._save()
            return entry

    def get_entry(self, entry_id: str) -> CatalogEntry | None:
        return self._entries.get(entry_id)

    def update_entry(self, entry: CatalogEntry) -> CatalogEntry:
        with self._lock:
            if entry.entry_id not in self._entries:
                raise EntryNotFoundError(f"Entry not found: {entry.entry_id}")
            self._entries[entry.entry_id] = entry
            self._save()
            return entry

    def remove_entry(self, entry_id: str) -> bool:
        with self._lock:
            if entry_id not in self._entries:
                return False
            del self._entries[entry_id]
            self._save()
            return True

    def list_entries(
        self,
        status: str | None = None,
        source_type: str | None = None,
        domain: str | None = None,
    ) -> list[CatalogEntry]:
        results = list(self._entries.values())
        if status:
            results = [e for e in results if e.status == status]
        if source_type:
            results = [e for e in results if e.source_type == source_type]
        if domain:
            results = [e for e in results if e.domain == domain]
        return results

    def search_entries(self, query: str) -> list[CatalogEntry]:
        q = query.lower()
        return [
            e
            for e in self._entries.values()
            if q in e.name.lower()
            or q in e.domain.lower()
            or any(q in t.lower() for t in e.tags)
            or q in e.notes.lower()
        ]

    def count_entries(self) -> int:
        return len(self._entries)
