"""History manager — persists training history for plotting and analysis.

Stores per-epoch history entries with loss, metrics, learning rate,
and checkpoint paths. Provides dict-formatted output for direct plotting.
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.interfaces import HistoryManagerInterface
from backend.training.models import HistoryEntry

logger = logging.getLogger("dss.training.history")


class HistoryManager(HistoryManagerInterface):
    """Persists training history as JSON files.

    Each experiment gets a single history file containing an ordered
    list of epoch entries.
    """

    def __init__(self, history_dir: Path | None = None) -> None:
        self._config = training_config
        self._history_dir = history_dir or self._config.history_dir
        self._history_dir.mkdir(parents=True, exist_ok=True)

    def record_entry(self, entry: HistoryEntry) -> HistoryEntry:
        logger.debug("History entry recorded: %s epoch %d", entry.experiment_id, entry.epoch)
        entries = self._load_all(entry.experiment_id)
        entries.append(entry)
        self._persist(entry.experiment_id, entries)
        return entry

    def get_history(self, experiment_id: str) -> list[HistoryEntry]:
        return self._load_all(experiment_id)

    def get_latest_entry(self, experiment_id: str) -> HistoryEntry | None:
        entries = self._load_all(experiment_id)
        return entries[-1] if entries else None

    def get_history_as_dicts(self, experiment_id: str) -> list[dict[str, object]]:
        return [entry.model_dump() for entry in self._load_all(experiment_id)]

    def _history_path(self, experiment_id: str) -> Path:
        return self._history_dir / f"{experiment_id}_history.json"

    def _load_all(self, experiment_id: str) -> list[HistoryEntry]:
        path = self._history_path(experiment_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return [HistoryEntry(**item) for item in data]
        except Exception:
            return []

    def _persist(self, experiment_id: str, entries: list[HistoryEntry]) -> None:
        path = self._history_path(experiment_id)
        path.write_text(
            json.dumps([e.model_dump() for e in entries], indent=2, default=str),
        )
