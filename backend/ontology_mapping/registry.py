"""In-memory registry of dataset mapping rules.

Thread-safe by virtue of asyncio single-threaded access.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.ontology_mapping.exceptions import (
    DatasetAlreadyRegisteredError,
    DatasetNotFoundError,
)
from backend.ontology_mapping.models import MappingRule, MappingVersion


class MappingRegistry:
    """Stores and retrieves mapping rules by dataset.

    Supports:
    - register_dataset / remove_dataset
    - register_rule / remove_rule / lookup
    - versioning and history
    """

    def __init__(self, version: str = "1.0.0") -> None:
        self._version: str = version
        self._datasets: dict[str, list[MappingRule]] = {}
        self._index: dict[str, dict[str, list[MappingRule]]] = {}
        self._history: dict[str, list[MappingVersion]] = {}
        self._ontology_version: str = "1.0.0"

    @property
    def ontology_version(self) -> str:
        return self._ontology_version

    @ontology_version.setter
    def ontology_version(self, value: str) -> None:
        self._ontology_version = value

    async def register_dataset(self, dataset_name: str) -> None:
        if dataset_name in self._datasets:
            raise DatasetAlreadyRegisteredError(
                f"Dataset '{dataset_name}' is already registered"
            )
        self._datasets[dataset_name] = []
        self._index[dataset_name] = {}
        self._history[dataset_name] = []

    async def remove_dataset(self, dataset_name: str) -> None:
        self._require_dataset(dataset_name)
        del self._datasets[dataset_name]
        del self._index[dataset_name]
        del self._history[dataset_name]

    async def register_rule(self, rule: MappingRule) -> None:
        self._require_dataset(rule.dataset_name)
        if rule.dataset_name not in self._index:
            self._index[rule.dataset_name] = {}
        label_index = self._index[rule.dataset_name]
        label_lower = rule.source_label.lower()
        if label_lower not in label_index:
            label_index[label_lower] = []
        label_index[label_lower].append(rule)
        self._datasets[rule.dataset_name].append(rule)

    async def remove_rule(self, rule_id: str) -> None:
        for dataset_name in list(self._datasets):
            rules = self._datasets[dataset_name]
            self._datasets[dataset_name] = [r for r in rules if r.rule_id != rule_id]
            label_index = self._index.get(dataset_name, {})
            for label_lower in list(label_index):
                label_index[label_lower] = [
                    r for r in label_index[label_lower] if r.rule_id != rule_id
                ]
                if not label_index[label_lower]:
                    del label_index[label_lower]

    async def lookup(
        self,
        dataset_name: str,
        label: str,
    ) -> list[MappingRule]:
        self._require_dataset(dataset_name)
        label_index = self._index.get(dataset_name, {})
        return list(label_index.get(label.lower(), []))

    async def get_rules(
        self,
        dataset_name: str,
    ) -> list[MappingRule]:
        self._require_dataset(dataset_name)
        return list(self._datasets[dataset_name])

    async def get_datasets(self) -> list[str]:
        return list(self._datasets.keys())

    async def version(
        self,
        dataset_name: str,
    ) -> MappingVersion | None:
        history = self._history.get(dataset_name)
        if not history:
            return None
        return history[-1]

    async def history(
        self,
        dataset_name: str,
    ) -> list[MappingVersion]:
        return list(self._history.get(dataset_name, []))

    async def commit_version(
        self,
        dataset_name: str,
        changelog: str = "",
    ) -> MappingVersion:
        self._require_dataset(dataset_name)
        rule_count = len(self._datasets[dataset_name])
        v = MappingVersion(
            version=self._version,
            ontology_version=self._ontology_version,
            created_at=datetime.now(timezone.utc),
            changelog=changelog,
            dataset_count=len(self._datasets),
            rule_count=rule_count,
        )
        self._history[dataset_name].append(v)
        return v

    async def clear(self) -> None:
        self._datasets.clear()
        self._index.clear()
        self._history.clear()

    def _require_dataset(self, dataset_name: str) -> None:
        if dataset_name not in self._datasets:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_name}' is not registered"
            )
