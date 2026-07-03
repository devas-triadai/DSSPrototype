"""Ontology data models.

All dataclasses are frozen for immutability and strong typing.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OntologyEntry:
    """A single concept in the ontology vocabulary."""

    canonical_name: str
    aliases: list[str] = field(default_factory=list)
    parent_category: str | None = None
    child_categories: list[str] = field(default_factory=list)
    civilian_equivalents: list[str] = field(default_factory=list)
    military_equivalents: list[str] = field(default_factory=list)
    description: str = ""
    confidence_base: float = 0.5


@dataclass(frozen=True)
class SemanticConcept:
    """A concept derived from ontology processing with confidence."""

    canonical_name: str
    confidence: float
    source: str  # "direct", "synonym", "category", "alias",
                  # "military_equivalent", "civilian_equivalent"
    parent_concepts: list[str] = field(default_factory=list)
    child_concepts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OntologyResult:
    """The complete output of ontology processing for a detector label."""

    original_label: str
    canonical_concept: str | None
    expanded_concepts: list[SemanticConcept]
    adjusted_confidence: float
    ontology_version: str
    mapping_path: list[str] = field(
        default_factory=list
    )  # e.g. ["truck", "wheeled_vehicle", "ground_vehicle"]

    def get_all_search_terms(self) -> list[str]:
        """Return all terms that should be used for knowledge base search."""
        terms = [self.original_label]
        if self.canonical_concept and self.canonical_concept != self.original_label:
            terms.append(self.canonical_concept)
        for concept in self.expanded_concepts:
            if concept.canonical_name not in terms:
                terms.append(concept.canonical_name)
        return terms


@dataclass(frozen=True)
class OntologyDataset:
    """A loaded ontology domain file."""

    version: str
    schema_version: str
    domain: str
    description: str
    entries: dict[str, OntologyEntry]

    def get_entry(self, canonical_name: str) -> OntologyEntry | None:
        return self.entries.get(canonical_name)

    def find_by_alias(self, alias: str) -> OntologyEntry | None:
        alias_lower = alias.lower().replace("_", " ").replace("-", " ")
        for entry in self.entries.values():
            if entry.canonical_name.lower() == alias_lower:
                return entry
            for a in entry.aliases:
                if a.lower() == alias_lower:
                    return entry
        return None


@dataclass(frozen=True)
class MappingResult:
    """Internal result of a single mapping operation."""

    canonical_name: str | None
    confidence: float
    source: str
