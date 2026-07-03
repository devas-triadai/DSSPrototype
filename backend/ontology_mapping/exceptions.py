"""Exception hierarchy for the Ontology Mapping Layer.

All mapping errors inherit from MappingError.
"""


class MappingError(RuntimeError):
    """Base exception for all ontology mapping errors."""


class RegistryError(MappingError):
    """Raised when a registry operation fails."""


class DatasetAlreadyRegisteredError(RegistryError):
    """Raised when attempting to register a dataset that already exists."""


class DatasetNotFoundError(RegistryError):
    """Raised when a dataset is not found in the registry."""


class MappingRuleError(MappingError):
    """Raised when a mapping rule is invalid."""


class OntologyResolverError(MappingError):
    """Raised when an ontology resolution operation fails."""


class OntologyNodeNotFoundError(OntologyResolverError):
    """Raised when a queried ontology node does not exist."""


class ConflictError(MappingError):
    """Raised when a mapping conflict cannot be resolved automatically."""


class CircularMappingError(ConflictError):
    """Raised when a circular mapping is detected."""


class ValidationError(MappingError):
    """Raised when ontology validation fails."""


class ExportError(MappingError):
    """Raised when an export operation fails."""


class VersionError(MappingError):
    """Raised when a version operation fails."""


class StatisticsError(MappingError):
    """Raised when statistics generation fails."""


class ConfigError(MappingError):
    """Raised when configuration is invalid."""
