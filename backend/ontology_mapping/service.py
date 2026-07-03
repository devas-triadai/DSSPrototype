"""Public API for the Ontology Mapping Layer.

Orchestrates all components behind a clean facade.
Every method is async: all public operations expect await.
"""

from __future__ import annotations

from backend.ontology_mapping.config import (
    OntologyMappingConfig,
    ontology_mapping_config,
)
from backend.ontology_mapping.conflict_resolver import ConflictResolver
from backend.ontology_mapping.dataset_mapper import DatasetMapper
from backend.ontology_mapping.exceptions import (
    DatasetNotFoundError,
)
from backend.ontology_mapping.exporter import MappingExporter
from backend.ontology_mapping.mapping_engine import MappingEngine
from backend.ontology_mapping.mapping_statistics import (
    MappingStatisticsGenerator,
)
from backend.ontology_mapping.mapping_validator import MappingValidator
from backend.ontology_mapping.mapping_version import MappingVersionManager
from backend.ontology_mapping.models import (
    ConflictResolution,
    DatasetMapping,
    DatasetProfile,
    ExportFormat,
    MappingConflict,
    MappingResult,
    MappingRule,
    MappingStatistics,
    OntologyResolution,
    ResolutionType,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver
from backend.ontology_mapping.registry import MappingRegistry


class OntologyMappingService:
    """Public facade for the Ontology Mapping Layer.

    Usage::

        service = OntologyMappingService()
        await service.register_dataset(profile, rules)
        result = await service.map_label("coco", "truck")
    """

    def __init__(
        self,
        registry: MappingRegistry | None = None,
        engine: MappingEngine | None = None,
        resolver: OntologyResolver | None = None,
        conflict_resolver: ConflictResolver | None = None,
        validator: MappingValidator | None = None,
        statistics: MappingStatisticsGenerator | None = None,
        exporter: MappingExporter | None = None,
        version_manager: MappingVersionManager | None = None,
        config: OntologyMappingConfig | None = None,
    ) -> None:
        self._resolver = resolver or OntologyResolver()
        self._registry = registry or MappingRegistry()
        self._engine = engine or MappingEngine(resolver=self._resolver)
        self._conflict_resolver = conflict_resolver or ConflictResolver(
            resolver=self._resolver
        )
        self._validator = validator or MappingValidator(
            resolver=self._resolver
        )
        self._statistics = statistics or MappingStatisticsGenerator(
            resolver=self._resolver
        )
        self._exporter = exporter or MappingExporter()
        self._version_manager = version_manager or MappingVersionManager()
        self._config = config or ontology_mapping_config
        self._dataset_mapper = DatasetMapper(
            registry=self._registry,
            engine=self._engine,
        )

    async def register_dataset(
        self,
        profile: DatasetProfile,
        rules: list[MappingRule],
    ) -> DatasetMapping:
        return await self._dataset_mapper.register_dataset(profile, rules)

    async def map_dataset(
        self,
        dataset_name: str,
        labels: list[str],
    ) -> list[MappingResult]:
        return await self._dataset_mapper.map_labels(dataset_name, labels)

    async def map_label(
        self,
        dataset_name: str,
        label: str,
    ) -> MappingResult:
        return await self._dataset_mapper.map_label(dataset_name, label)

    async def validate(
        self,
        dataset_name: str | None = None,
    ) -> list[str]:
        errors: list[str] = []

        if self._config.validation_enabled:
            onto_errors = await self._validator.validate_ontology()
            errors.extend(onto_errors)

        if dataset_name is not None:
            try:
                rules = await self._registry.get_rules(dataset_name)
            except DatasetNotFoundError:
                errors.append(f"Dataset '{dataset_name}' is not registered")
                return errors

            mapping_errors = await self._validator.validate_mapping(rules)
            errors.extend(mapping_errors)

            rule_errors = await self._validator.validate_rules(rules)
            errors.extend(rule_errors)

        return errors

    async def resolve(self, value: str) -> OntologyResolution:
        return await self._resolver.resolve(value)

    async def statistics(
        self,
        dataset_name: str,
    ) -> MappingStatistics:
        rules = await self._registry.get_rules(dataset_name)
        return await self._statistics.compute(dataset_name, rules)

    async def export(
        self,
        dataset_name: str,
        fmt: str = "json",
    ) -> str:
        mapping = await self._dataset_mapper.get_mapping(dataset_name)
        if mapping is None:
            raise DatasetNotFoundError(
                f"Dataset '{dataset_name}' is not registered"
            )
        export_fmt = ExportFormat(fmt.lower())
        return await self._exporter.export(mapping, export_fmt)

    async def detect_conflicts(
        self,
        dataset_name: str,
    ) -> list[MappingConflict]:
        rules = await self._registry.get_rules(dataset_name)
        return await self._conflict_resolver.detect(rules)

    async def resolve_conflicts(
        self,
        dataset_name: str,
        strategy: ResolutionType = ResolutionType.HIGHEST_CONFIDENCE,
    ) -> list[ConflictResolution]:
        rules = await self._registry.get_rules(dataset_name)
        conflicts = await self._conflict_resolver.detect(rules)
        return await self._conflict_resolver.resolve_all(conflicts)

    async def coverage_report(
        self,
        dataset_name: str,
    ) -> str:
        stats = await self.statistics(dataset_name)
        return await self._statistics.coverage_report(stats)

    async def compatibility_report(
        self,
        dataset_a: str,
        dataset_b: str,
    ) -> str:
        stats_a = await self.statistics(dataset_a)
        stats_b = await self.statistics(dataset_b)
        return await self._statistics.compatibility_report(stats_a, stats_b)

    async def get_datasets(self) -> list[str]:
        return await self._registry.get_datasets()

    async def get_mapping(
        self,
        dataset_name: str,
    ) -> DatasetMapping | None:
        return await self._dataset_mapper.get_mapping(dataset_name)

    async def remove_dataset(self, dataset_name: str) -> None:
        await self._dataset_mapper.remove_dataset(dataset_name)

    async def register_alias(
        self,
        canonical_value: str,
        alias: str,
    ) -> None:
        self._engine.register_alias(canonical_value, alias)

    async def register_synonym(
        self,
        canonical_value: str,
        synonym: str,
    ) -> None:
        self._engine.register_synonym(canonical_value, synonym)

    async def register_regex(
        self,
        pattern: str,
        canonical_value: str,
        confidence: float = 0.7,
    ) -> None:
        self._engine.register_regex(pattern, canonical_value, confidence)

    async def validate_mapping_rules(
        self,
        rules: list[MappingRule],
    ) -> list[str]:
        return await self._validator.validate_rules(rules)
