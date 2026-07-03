"""Ontology Mapping Layer — translates external dataset labels into the canonical DSS ontology.

This package is COMPLETELY INDEPENDENT from all other DSS modules.
It depends only on:
  - ``backend.contracts.enums.core.ObjectType`` (read-only, never modified)
  - Standard library and common third-party packages

FROZEN ARCHITECTURE: No existing module, contract, API, runtime, or ontology
is modified by this package.
"""

# Config
from backend.ontology_mapping.config import (
    OntologyMappingConfig,
    ontology_mapping_config,
)
from backend.ontology_mapping.conflict_resolver import ConflictResolver
from backend.ontology_mapping.dataset_mapper import DatasetMapper

# Exceptions
from backend.ontology_mapping.exceptions import (
    CircularMappingError,
    ConfigError,
    ConflictError,
    DatasetAlreadyRegisteredError,
    DatasetNotFoundError,
    ExportError,
    MappingError,
    MappingRuleError,
    OntologyNodeNotFoundError,
    OntologyResolverError,
    RegistryError,
    StatisticsError,
    ValidationError,
    VersionError,
)
from backend.ontology_mapping.exporter import MappingExporter

# Interfaces
from backend.ontology_mapping.interfaces import (
    ConflictResolverInterface,
    DatasetMapperInterface,
    ExportInterface,
    MappingServiceInterface,
    OntologyMapperInterface,
    OntologyResolverInterface,
    RegistryInterface,
    StatisticsInterface,
    ValidationInterface,
)
from backend.ontology_mapping.mapping_engine import MappingEngine
from backend.ontology_mapping.mapping_statistics import (
    MappingStatisticsGenerator,
)
from backend.ontology_mapping.mapping_validator import MappingValidator
from backend.ontology_mapping.mapping_version import MappingVersionManager

# Models
from backend.ontology_mapping.models import (
    ConflictResolution,
    ConflictType,
    DatasetLabel,
    DatasetMapping,
    DatasetProfile,
    ExportFormat,
    MappingConflict,
    MappingResult,
    MappingRule,
    MappingStatistics,
    MappingVersion,
    MatchType,
    OntologyNode,
    OntologyResolution,
    ResolutionType,
)
from backend.ontology_mapping.ontology_resolver import OntologyResolver

# Core components
from backend.ontology_mapping.registry import MappingRegistry

# Service
from backend.ontology_mapping.service import OntologyMappingService

__all__ = [
    # Config
    "OntologyMappingConfig",
    "ontology_mapping_config",
    # Exceptions
    "MappingError",
    "RegistryError",
    "DatasetAlreadyRegisteredError",
    "DatasetNotFoundError",
    "MappingRuleError",
    "OntologyResolverError",
    "OntologyNodeNotFoundError",
    "ConflictError",
    "CircularMappingError",
    "ValidationError",
    "ExportError",
    "VersionError",
    "StatisticsError",
    "ConfigError",
    # Enums
    "MatchType",
    "ConflictType",
    "ResolutionType",
    "ExportFormat",
    # Models
    "OntologyNode",
    "DatasetLabel",
    "MappingRule",
    "MappingConflict",
    "MappingResult",
    "DatasetMapping",
    "DatasetProfile",
    "MappingStatistics",
    "MappingVersion",
    "OntologyResolution",
    "ConflictResolution",
    # Interfaces
    "OntologyMapperInterface",
    "DatasetMapperInterface",
    "OntologyResolverInterface",
    "ConflictResolverInterface",
    "ExportInterface",
    "ValidationInterface",
    "StatisticsInterface",
    "RegistryInterface",
    "MappingServiceInterface",
    # Core components
    "MappingRegistry",
    "MappingEngine",
    "OntologyResolver",
    "ConflictResolver",
    "MappingValidator",
    "MappingStatisticsGenerator",
    "MappingVersionManager",
    "DatasetMapper",
    "MappingExporter",
    # Service
    "OntologyMappingService",
]
