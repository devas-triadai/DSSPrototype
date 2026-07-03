"""Dataset-level mapping orchestration.

Coordinates registration, rule management, mapping, and cleanup
for a single external dataset.
"""

from __future__ import annotations

from datetime import datetime, timezone

from backend.ontology_mapping.mapping_engine import MappingEngine
from backend.ontology_mapping.models import (
    DatasetMapping,
    DatasetProfile,
    MappingResult,
    MappingRule,
)
from backend.ontology_mapping.registry import MappingRegistry


class DatasetMapper:
    """Manages the full mapping lifecycle for one dataset.

    Coordinates between the registry (persistence), mapping engine
    (translation), and resolver (ontology validation).
    """

    def __init__(
        self,
        registry: MappingRegistry | None = None,
        engine: MappingEngine | None = None,
        ontology_version: str = "1.0.0",
    ) -> None:
        self._registry = registry or MappingRegistry()
        self._engine = engine or MappingEngine()
        self._ontology_version = ontology_version
        self._profiles: dict[str, DatasetProfile] = {}

    async def register_dataset(
        self,
        profile: DatasetProfile,
        rules: list[MappingRule],
    ) -> DatasetMapping:
        if profile.dataset_name in self._profiles:
            existing = await self._get_mapping_internal(
                profile.dataset_name
            )
            if existing is not None:
                return existing

        await self._registry.register_dataset(profile.dataset_name)
        self._profiles[profile.dataset_name] = profile

        for rule in rules:
            await self._registry.register_rule(rule)

        await self._registry.commit_version(
            profile.dataset_name,
            f"Initial mapping for dataset '{profile.dataset_name}'",
        )

        stored_rules = await self._registry.get_rules(
            profile.dataset_name
        )
        return DatasetMapping(
            dataset_name=profile.dataset_name,
            dataset_version=profile.version,
            ontology_version=self._ontology_version,
            rules=tuple(stored_rules),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

    async def get_mapping(
        self,
        dataset_name: str,
    ) -> DatasetMapping | None:
        return await self._get_mapping_internal(dataset_name)

    async def remove_dataset(
        self,
        dataset_name: str,
    ) -> None:
        self._profiles.pop(dataset_name, None)
        await self._registry.remove_dataset(dataset_name)

    async def map_labels(
        self,
        dataset_name: str,
        labels: list[str],
    ) -> list[MappingResult]:
        rules = await self._registry.get_rules(dataset_name)
        return await self._engine.map_batch(
            dataset_name, labels, rules
        )

    async def map_label(
        self,
        dataset_name: str,
        label: str,
    ) -> MappingResult:
        rules = await self._registry.get_rules(dataset_name)
        return await self._engine.map_label(
            dataset_name, label, rules
        )

    async def _get_mapping_internal(
        self,
        dataset_name: str,
    ) -> DatasetMapping | None:
        profile = self._profiles.get(dataset_name)
        if profile is None:
            try:
                rules = await self._registry.get_rules(dataset_name)
            except Exception:
                return None
            return DatasetMapping(
                dataset_name=dataset_name,
                dataset_version="unknown",
                ontology_version=self._ontology_version,
                rules=tuple(rules),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        rules = await self._registry.get_rules(dataset_name)
        return DatasetMapping(
            dataset_name=dataset_name,
            dataset_version=profile.version,
            ontology_version=self._ontology_version,
            rules=tuple(rules),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
