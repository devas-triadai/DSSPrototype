"""Abstract interfaces for the Ontology Mapping Layer.

Every public capability is defined here as an ABC.
Concrete implementations live in sibling modules.
"""

from abc import ABC, abstractmethod

from backend.ontology_mapping.models import (
    ConflictResolution,
    DatasetMapping,
    DatasetProfile,
    MappingConflict,
    MappingResult,
    MappingRule,
    MappingStatistics,
    MappingVersion,
    OntologyNode,
    OntologyResolution,
)


class OntologyMapperInterface(ABC):
    """Translates a dataset label into a canonical ontology value."""

    @abstractmethod
    async def map_label(
        self,
        dataset_name: str,
        label: str,
    ) -> MappingResult:
        ...

    @abstractmethod
    async def map_batch(
        self,
        dataset_name: str,
        labels: list[str],
    ) -> list[MappingResult]:
        ...


class DatasetMapperInterface(ABC):
    """Manages the full mapping lifecycle for a single dataset."""

    @abstractmethod
    async def register_dataset(
        self,
        profile: DatasetProfile,
    ) -> DatasetMapping:
        ...

    @abstractmethod
    async def get_mapping(
        self,
        dataset_name: str,
    ) -> DatasetMapping | None:
        ...

    @abstractmethod
    async def remove_dataset(
        self,
        dataset_name: str,
    ) -> None:
        ...


class OntologyResolverInterface(ABC):
    """Navigates the ontology tree."""

    @abstractmethod
    async def resolve(
        self,
        value: str,
    ) -> OntologyResolution:
        ...

    @abstractmethod
    async def get_node(
        self,
        value: str,
    ) -> OntologyNode | None:
        ...

    @abstractmethod
    async def get_children(
        self,
        value: str,
    ) -> list[OntologyNode]:
        ...

    @abstractmethod
    async def get_parent(
        self,
        value: str,
    ) -> OntologyNode | None:
        ...

    @abstractmethod
    async def get_ancestors(
        self,
        value: str,
    ) -> list[OntologyNode]:
        ...

    @abstractmethod
    async def get_siblings(
        self,
        value: str,
    ) -> list[OntologyNode]:
        ...

    @abstractmethod
    async def get_leaves(
        self,
    ) -> list[OntologyNode]:
        ...

    @abstractmethod
    async def contains(
        self,
        value: str,
    ) -> bool:
        ...


class ConflictResolverInterface(ABC):
    """Detects and resolves mapping conflicts."""

    @abstractmethod
    async def detect(
        self,
        rules: list[MappingRule],
    ) -> list[MappingConflict]:
        ...

    @abstractmethod
    async def resolve(
        self,
        conflict: MappingConflict,
    ) -> ConflictResolution:
        ...

    @abstractmethod
    async def resolve_all(
        self,
        conflicts: list[MappingConflict],
    ) -> list[ConflictResolution]:
        ...


class ExportInterface(ABC):
    """Exports mappings to serialized formats."""

    @abstractmethod
    async def to_json(
        self,
        mapping: DatasetMapping,
    ) -> str:
        ...

    @abstractmethod
    async def to_yaml(
        self,
        mapping: DatasetMapping,
    ) -> str:
        ...

    @abstractmethod
    async def to_csv(
        self,
        mapping: DatasetMapping,
    ) -> str:
        ...

    @abstractmethod
    async def export_all(
        self,
        mapping: DatasetMapping,
        directory: str,
    ) -> dict[str, str]:
        ...


class ValidationInterface(ABC):
    """Validates ontology structure and mapping rules."""

    @abstractmethod
    async def validate_ontology(
        self,
    ) -> list[str]:
        ...

    @abstractmethod
    async def validate_mapping(
        self,
        rules: list[MappingRule],
    ) -> list[str]:
        ...

    @abstractmethod
    async def validate_rules(
        self,
        rules: list[MappingRule],
    ) -> list[str]:
        ...


class StatisticsInterface(ABC):
    """Generates mapping statistics and reports."""

    @abstractmethod
    async def compute(
        self,
        dataset_name: str,
        rules: list[MappingRule],
    ) -> MappingStatistics:
        ...

    @abstractmethod
    async def coverage_report(
        self,
        statistics: MappingStatistics,
    ) -> str:
        ...

    @abstractmethod
    async def compatibility_report(
        self,
        source: MappingStatistics,
        target: MappingStatistics,
    ) -> str:
        ...


class RegistryInterface(ABC):
    """Stores and retrieves mapping rules."""

    @abstractmethod
    async def register_dataset(
        self,
        dataset_name: str,
    ) -> None:
        ...

    @abstractmethod
    async def register_rule(
        self,
        rule: MappingRule,
    ) -> None:
        ...

    @abstractmethod
    async def remove_rule(
        self,
        rule_id: str,
    ) -> None:
        ...

    @abstractmethod
    async def lookup(
        self,
        dataset_name: str,
        label: str,
    ) -> list[MappingRule]:
        ...

    @abstractmethod
    async def get_rules(
        self,
        dataset_name: str,
    ) -> list[MappingRule]:
        ...

    @abstractmethod
    async def get_datasets(
        self,
    ) -> list[str]:
        ...

    @abstractmethod
    async def version(
        self,
        dataset_name: str,
    ) -> MappingVersion | None:
        ...

    @abstractmethod
    async def history(
        self,
        dataset_name: str,
    ) -> list[MappingVersion]:
        ...


class MappingServiceInterface(ABC):
    """Public API for the Ontology Mapping Layer."""

    @abstractmethod
    async def register_dataset(
        self,
        profile: DatasetProfile,
        rules: list[MappingRule],
    ) -> DatasetMapping:
        ...

    @abstractmethod
    async def map_dataset(
        self,
        dataset_name: str,
        labels: list[str],
    ) -> list[MappingResult]:
        ...

    @abstractmethod
    async def map_label(
        self,
        dataset_name: str,
        label: str,
    ) -> MappingResult:
        ...

    @abstractmethod
    async def validate(
        self,
        dataset_name: str,
    ) -> list[str]:
        ...

    @abstractmethod
    async def resolve(
        self,
        value: str,
    ) -> OntologyResolution:
        ...

    @abstractmethod
    async def statistics(
        self,
        dataset_name: str,
    ) -> MappingStatistics:
        ...

    @abstractmethod
    async def export(
        self,
        dataset_name: str,
        fmt: str,
    ) -> str:
        ...

    @abstractmethod
    async def detect_conflicts(
        self,
        dataset_name: str,
    ) -> list[MappingConflict]:
        ...

    @abstractmethod
    async def resolve_conflicts(
        self,
        dataset_name: str,
    ) -> list[ConflictResolution]:
        ...
