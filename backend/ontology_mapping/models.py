"""Immutable data models for the Ontology Mapping Layer.

Every model is a frozen Pydantic dataclass with validated fields.
No business logic — data carriers only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class MatchType(str, Enum):
    """Strategy used to match a dataset label to an ontology node."""

    EXACT = "exact"
    ALIAS = "alias"
    CASE_INSENSITIVE = "case_insensitive"
    PLURAL = "plural"
    SYNONYM = "synonym"
    REGEX = "regex"
    EMBEDDING = "embedding"


class ConflictType(str, Enum):
    """Category of mapping conflict."""

    DUPLICATE = "duplicate"
    CONFLICTING = "conflicting"
    CIRCULAR = "circular"
    MISSING_NODE = "missing_node"
    AMBIGUOUS_ALIAS = "ambiguous_alias"
    UNKNOWN_LABEL = "unknown_label"


class ResolutionType(str, Enum):
    """Strategy used to resolve a conflict."""

    FIRST_WINS = "first_wins"
    LAST_WINS = "last_wins"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MERGE = "merge"
    MANUAL = "manual"


class ExportFormat(str, Enum):
    """Supported export serialization formats."""

    JSON = "json"
    YAML = "yaml"
    CSV = "csv"


class OntologyNode(BaseModel):
    """A single node in the DSS ontology tree.

    Nodes form a rooted tree rooted at ``root`` with depth 0.
    """

    model_config = ConfigDict(frozen=True)

    value: str = Field(
        ...,
        min_length=1,
        description="Dotted ontology path, e.g. 'ground_vehicle.car'",
    )
    name: str = Field(
        ...,
        min_length=1,
        description="Human-readable class name, e.g. 'Car'",
    )
    category: str = Field(
        ...,
        min_length=1,
        description="Top-level category key, e.g. 'ground_vehicle'",
    )
    category_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable category name, e.g. 'Ground Vehicle'",
    )
    parent: str | None = Field(
        None,
        description="Value of the parent node, or None for root",
    )
    children: frozenset[str] = Field(
        default_factory=frozenset,
        description="Values of direct child nodes",
    )
    depth: int = Field(
        ...,
        ge=0,
        le=4,
        description="Tree depth (0=root, 1=category, 2=class, 3=subtype)",
    )
    is_leaf: bool = Field(
        ...,
        description="True if this node has no children",
    )
    enum_member_name: str | None = Field(
        None,
        description="Corresponding ObjectType enum member name, if any",
    )


class DatasetLabel(BaseModel):
    """A single label from an external dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., min_length=1, description="Source dataset name")
    original_label: str = Field(..., min_length=1, description="Raw label text")
    normalized_label: str | None = Field(
        None,
        description="Case/whitespace-normalized form",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence that this label is valid",
    )


class MappingRule(BaseModel):
    """Maps one or more external dataset labels to a canonical ontology node."""

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique rule identifier",
    )
    dataset_name: str = Field(..., min_length=1, description="Source dataset name")
    source_label: str = Field(
        ...,
        min_length=1,
        description="Original label from the external dataset",
    )
    canonical_value: str = Field(
        ...,
        min_length=1,
        description="Target ontology dotted value, e.g. 'ground_vehicle.car'",
    )
    match_type: MatchType = Field(
        ...,
        description="Matching strategy used",
    )
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this mapping (1.0 = exact)",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this rule was created",
    )
    version: str = Field(
        "1.0.0",
        min_length=1,
        description="Mapping layer version when created",
    )


class MappingConflict(BaseModel):
    """A detected conflict between mapping rules."""

    model_config = ConfigDict(frozen=True)

    conflict_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique conflict identifier",
    )
    conflict_type: ConflictType = Field(
        ...,
        description="Category of conflict",
    )
    source_label: str = Field(
        ...,
        min_length=1,
        description="The label that causes the conflict",
    )
    rules: tuple[MappingRule, ...] = Field(
        ...,
        min_length=1,
        description="Conflicting rules, ordered by creation time",
    )
    description: str = Field(
        ...,
        min_length=1,
        description="Human-readable explanation of the conflict",
    )
    resolved: bool = Field(
        False,
        description="Whether this conflict has been resolved",
    )
    resolution: str | None = Field(
        None,
        description="How the conflict was resolved",
    )


class MappingResult(BaseModel):
    """The result of mapping a single dataset label to the ontology."""

    model_config = ConfigDict(frozen=True)

    source_label: str = Field(
        ...,
        min_length=1,
        description="Original label from the external dataset",
    )
    canonical_value: str = Field(
        ...,
        min_length=1,
        description="Resolved ontology dotted value",
    )
    canonical_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable ontology class name",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence of this mapping result",
    )
    match_type: MatchType = Field(
        ...,
        description="Matching strategy that produced this result",
    )
    rule_id: str | None = Field(
        None,
        description="Rule that produced this result, if any",
    )
    alternatives: tuple[tuple[str, float], ...] = Field(
        default_factory=tuple,
        description="Alternative mappings with their confidence scores",
    )


class DatasetMapping(BaseModel):
    """Complete mapping state for a single dataset."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., min_length=1, description="Source dataset name")
    dataset_version: str = Field(
        ...,
        min_length=1,
        description="Version of the external dataset",
    )
    ontology_version: str = Field(
        ...,
        min_length=1,
        description="Version of the ontology at mapping time",
    )
    rules: tuple[MappingRule, ...] = Field(
        default_factory=tuple,
        description="All registered mapping rules",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this mapping was created",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this mapping was last updated",
    )


class DatasetProfile(BaseModel):
    """Metadata about an external dataset relevant to ontology mapping."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., min_length=1, description="Unique dataset name")
    version: str = Field(
        "1.0.0",
        min_length=1,
        description="Dataset version string",
    )
    label_count: int = Field(..., ge=1, description="Number of distinct labels")
    labels: tuple[str, ...] = Field(
        ...,
        min_length=1,
        description="All distinct label strings in the dataset",
    )
    description: str | None = Field(
        None,
        description="Human-readable dataset description",
    )


class MappingStatistics(BaseModel):
    """Statistics about a dataset's ontology mapping."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = Field(..., min_length=1, description="Source dataset name")
    total_labels: int = Field(..., ge=0, description="Total unique labels")
    mapped_labels: int = Field(
        ...,
        ge=0,
        description="Labels successfully mapped to ontology",
    )
    ignored_labels: int = Field(
        ...,
        ge=0,
        description="Labels that were intentionally ignored",
    )
    unknown_labels: int = Field(
        ...,
        ge=0,
        description="Labels that could not be mapped",
    )
    coverage_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of labels successfully mapped",
    )
    conflict_count: int = Field(..., ge=0, description="Number of active conflicts")
    duplicate_alias_count: int = Field(
        ...,
        ge=0,
        description="Number of duplicate alias mappings",
    )
    ontology_version: str = Field(
        ...,
        min_length=1,
        description="Ontology version at computation time",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When these statistics were generated",
    )


class MappingVersion(BaseModel):
    """Version snapshot for a dataset's mapping state."""

    model_config = ConfigDict(frozen=True)

    version: str = Field(
        ...,
        min_length=1,
        description="Mapping layer version identifier",
    )
    ontology_version: str = Field(
        ...,
        min_length=1,
        description="Ontology version used",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this version was created",
    )
    changelog: str = Field(
        "",
        description="Description of changes in this version",
    )
    dataset_count: int = Field(
        0,
        ge=0,
        description="Number of registered datasets",
    )
    rule_count: int = Field(
        0,
        ge=0,
        description="Total mapping rules across all datasets",
    )


class OntologyResolution(BaseModel):
    """Result of an ontology resolution query."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(
        ...,
        min_length=1,
        description="The value that was queried",
    )
    resolved_node: OntologyNode | None = Field(
        ...,
        description="The node matching the query, or None",
    )
    parent: OntologyNode | None = Field(
        None,
        description="Parent node, if any",
    )
    children: tuple[OntologyNode, ...] = Field(
        default_factory=tuple,
        description="Direct child nodes",
    )
    ancestors: tuple[OntologyNode, ...] = Field(
        default_factory=tuple,
        description="All ancestor nodes (root-ward)",
    )
    siblings: tuple[OntologyNode, ...] = Field(
        default_factory=tuple,
        description="Sibling nodes (same parent, excluding self)",
    )
    path: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Full path from root to this node",
    )


class ConflictResolution(BaseModel):
    """Result of resolving a single mapping conflict."""

    model_config = ConfigDict(frozen=True)

    conflict_id: str = Field(
        ...,
        min_length=1,
        description="The conflict that was resolved",
    )
    resolution_type: ResolutionType = Field(
        ...,
        description="Strategy used for resolution",
    )
    chosen_rule: MappingRule | None = Field(
        None,
        description="The rule selected by resolution, if applicable",
    )
    merged_value: str | None = Field(
        None,
        description="Merged canonical value, if merge resolution was used",
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Why this resolution was chosen",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the resolution was applied",
    )
