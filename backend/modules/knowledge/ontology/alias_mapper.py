"""Alias Mapper.

Maintains alternate names and resolves raw labels to canonical concepts.
Also provides military and civilian equivalent mappings.
"""

import logging
from typing import Any

from backend.modules.knowledge.ontology.config import (
    CIVILIAN_EQUIVALENT_DECAY,
    MILITARY_EQUIVALENT_DECAY,
)
from backend.modules.knowledge.ontology.interfaces import AliasMapperInterface
from backend.modules.knowledge.ontology.models import OntologyDataset, SemanticConcept

logger = logging.getLogger("dss.ontology.alias_mapper")


class AliasMapper(AliasMapperInterface):
    """Resolves aliases and provides equivalent mappings."""

    def __init__(self, datasets: list[OntologyDataset]) -> None:
        self._datasets = datasets

    def get_aliases(self, canonical_name: str) -> list[str]:
        """Return all aliases for a canonical concept."""
        entry = self._find_entry(canonical_name)
        if entry is None:
            return []
        return list(entry.aliases)

    def resolve(self, raw_label: str) -> str | None:
        """Resolve a raw label to canonical name via alias lookup."""
        normalized = raw_label.lower().strip().replace("_", " ").replace("-", " ")
        for dataset in self._datasets:
            for entry in dataset.entries.values():
                if entry.canonical_name.lower() == normalized:
                    return entry.canonical_name
                for alias in entry.aliases:
                    if alias.lower() == normalized:
                        return entry.canonical_name
        return None

    def get_equivalent_concepts(
        self,
        canonical_name: str,
        base_confidence: float,
    ) -> list[SemanticConcept]:
        """Return military and civilian equivalents with decayed confidence."""
        concepts: list[SemanticConcept] = []
        entry = self._find_entry(canonical_name)
        if entry is None:
            return concepts

        # Military equivalents
        for eq in entry.military_equivalents:
            concepts.append(
                SemanticConcept(
                    canonical_name=eq,
                    confidence=round(base_confidence * MILITARY_EQUIVALENT_DECAY, 3),
                    source="military_equivalent",
                    parent_concepts=[],
                    child_concepts=[],
                )
            )
            logger.debug(
                "AliasMapper | '%s' -> military equivalent '%s' (confidence=%.3f)",
                canonical_name,
                eq,
                base_confidence * MILITARY_EQUIVALENT_DECAY,
            )

        # Civilian equivalents
        for eq in entry.civilian_equivalents:
            concepts.append(
                SemanticConcept(
                    canonical_name=eq,
                    confidence=round(base_confidence * CIVILIAN_EQUIVALENT_DECAY, 3),
                    source="civilian_equivalent",
                    parent_concepts=[],
                    child_concepts=[],
                )
            )
            logger.debug(
                "AliasMapper | '%s' -> civilian equivalent '%s' (confidence=%.3f)",
                canonical_name,
                eq,
                base_confidence * CIVILIAN_EQUIVALENT_DECAY,
            )

        return concepts

    def _find_entry(self, canonical_name: str) -> Any | None:
        for dataset in self._datasets:
            entry = dataset.get_entry(canonical_name)
            if entry:
                return entry
        return None
