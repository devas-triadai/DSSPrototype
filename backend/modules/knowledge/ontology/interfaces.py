"""Ontology module interfaces.

Defines abstract base classes for all ontology processing components.
All implementations are dependency-injected and stateless where possible.
"""

from abc import ABC, abstractmethod

from backend.modules.knowledge.ontology.models import (
    OntologyResult,
)


class SynonymMapperInterface(ABC):
    """Maps raw detector labels to canonical ontology concepts."""

    @abstractmethod
    def map(self, label: str) -> str | None:
        """Return the canonical name for a given raw label, or None if unknown."""


class CategoryMapperInterface(ABC):
    """Maps canonical concepts into hierarchical parent and child categories."""

    @abstractmethod
    def get_parents(self, canonical_name: str) -> list[str]:
        """Return ordered parent category names (closest first)."""

    @abstractmethod
    def get_children(self, canonical_name: str) -> list[str]:
        """Return child category names."""


class AliasMapperInterface(ABC):
    """Maintains alternate names and aliases for ontology concepts."""

    @abstractmethod
    def get_aliases(self, canonical_name: str) -> list[str]:
        """Return all registered aliases for a canonical concept."""

    @abstractmethod
    def resolve(self, raw_label: str) -> str | None:
        """Resolve a raw label to its canonical name via alias lookup."""


class ConfidenceAdjusterInterface(ABC):
    """Adjusts confidence scores based on ontology mapping quality."""

    @abstractmethod
    def adjust(
        self,
        detector_confidence: float,
        ontology_confidence: float,
        mapping_depth: int,
    ) -> float:
        """Return adjusted confidence that never exceeds detector confidence."""


class OntologyEngineInterface(ABC):
    """Coordinates all ontology processing into a single result."""

    @abstractmethod
    def process(
        self,
        label: str,
        detector_confidence: float,
    ) -> OntologyResult:
        """Transform a raw detector label into a structured ontology result."""

    @abstractmethod
    def expand_query(self, query: str) -> list[str]:
        """Expand a query into all related search terms (original + ontology)."""

    @abstractmethod
    def get_version(self) -> str:
        """Return the loaded ontology version string."""


class OntologyServiceInterface(ABC):
    """Service-level wrapper for ontology processing."""

    @abstractmethod
    def process(
        self,
        label: str,
        detector_confidence: float,
    ) -> OntologyResult:
        """Process a detector label through the ontology pipeline."""

    @abstractmethod
    def expand_query(self, query: str) -> list[str]:
        """Expand a query string into all related search terms."""

    @abstractmethod
    def get_version(self) -> str:
        """Return ontology version."""
