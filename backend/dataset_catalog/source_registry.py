"""JSON-backed source registry for tracking dataset sources."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from backend.dataset_catalog.config import dc_config
from backend.dataset_catalog.exceptions import SourceNotFoundError
from backend.dataset_catalog.interfaces import SourceRegistryInterface
from backend.dataset_catalog.models import SourceInfo

logger = logging.getLogger("dss.dataset_catalog.source_registry")


class SourceRegistry(SourceRegistryInterface):
    """JSON-file-backed persistent source registry.

    Thread-safe via Lock. Tracks source metadata, reliability scores,
    and fetch history.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path or dc_config.sources_db_path
        self._lock = Lock()
        self._sources: dict[str, SourceInfo] = {}
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
                self._sources = {
                    sid: SourceInfo(**data) for sid, data in raw.items()
                }
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Failed to load sources DB: %s — starting fresh", exc)
                self._sources = {}
        else:
            self._sources = {}

    def _save(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        raw = {
            sid: source.model_dump(mode="json")
            for sid, source in self._sources.items()
        }
        with self._db_path.open("w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def register_source(self, source: SourceInfo) -> SourceInfo:
        with self._lock:
            self._sources[source.source_id] = source
            self._save()
            return source

    def get_source(self, source_id: str) -> SourceInfo | None:
        return self._sources.get(source_id)

    def update_source(self, source: SourceInfo) -> SourceInfo:
        with self._lock:
            if source.source_id not in self._sources:
                raise SourceNotFoundError(f"Source not found: {source.source_id}")
            self._sources[source.source_id] = source
            self._save()
            return source

    def list_sources(self, source_type: str | None = None) -> list[SourceInfo]:
        if source_type:
            return [s for s in self._sources.values() if s.source_type == source_type]
        return list(self._sources.values())

    def record_success(self, source_id: str) -> None:
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                raise SourceNotFoundError(f"Source not found: {source_id}")
            updated = SourceInfo(
                source_id=source.source_id,
                name=source.name,
                source_type=source.source_type,
                url=source.url,
                description=source.description,
                reliability=self._compute_reliability(
                    source.successful_fetches + 1,
                    source.failed_fetches,
                ),
                total_fetches=source.total_fetches + 1,
                successful_fetches=source.successful_fetches + 1,
                failed_fetches=source.failed_fetches,
                last_fetch=datetime.now(timezone.utc).isoformat(),
                last_error="",
                tags=source.tags,
                created_at=source.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._sources[source_id] = updated
            self._save()

    def record_failure(self, source_id: str, error: str) -> None:
        with self._lock:
            source = self._sources.get(source_id)
            if source is None:
                raise SourceNotFoundError(f"Source not found: {source_id}")
            updated = SourceInfo(
                source_id=source.source_id,
                name=source.name,
                source_type=source.source_type,
                url=source.url,
                description=source.description,
                reliability=self._compute_reliability(
                    source.successful_fetches,
                    source.failed_fetches + 1,
                ),
                total_fetches=source.total_fetches + 1,
                successful_fetches=source.successful_fetches,
                failed_fetches=source.failed_fetches + 1,
                last_fetch=datetime.now(timezone.utc).isoformat(),
                last_error=error,
                tags=source.tags,
                created_at=source.created_at,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
            self._sources[source_id] = updated
            self._save()

    def get_reliability(self, source_id: str) -> float:
        source = self._sources.get(source_id)
        if source is None:
            raise SourceNotFoundError(f"Source not found: {source_id}")
        return source.reliability

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_reliability(success: int, failures: int) -> float:
        total = success + failures
        if total == 0:
            return 0.5
        ratio = success / total
        return min(max(ratio, dc_config.min_source_reliability), dc_config.max_source_reliability)
