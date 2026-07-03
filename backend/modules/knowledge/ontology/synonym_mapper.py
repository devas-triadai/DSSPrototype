"""Synonym Mapper.

Maps raw detector labels (e.g. "automobile", "uav") to canonical
ontology names (e.g. "car", "drone").
"""

import logging

from backend.modules.knowledge.ontology.interfaces import SynonymMapperInterface
from backend.modules.knowledge.ontology.models import (
    MappingResult,
    OntologyDataset,
    OntologyEntry,
)

logger = logging.getLogger("dss.ontology.synonym_mapper")


class SynonymMapper(SynonymMapperInterface):
    """Maps equivalent labels to canonical ontology concepts."""

    def __init__(self, datasets: list[OntologyDataset]) -> None:
        self._datasets = datasets
        self._alias_map: dict[str, str] = {}
        self._build_map()

    def _build_map(self) -> None:
        """Build a flat alias -> canonical_name lookup table."""
        for dataset in self._datasets:
            for entry in dataset.entries.values():
                # Canonical name maps to itself
                self._alias_map[entry.canonical_name.lower()] = entry.canonical_name
                # Aliases map to canonical
                for alias in entry.aliases:
                    self._alias_map[alias.lower()] = entry.canonical_name
                # Also index hyphenated and underscored variants
                canonical_lower = entry.canonical_name.lower()
                self._alias_map[canonical_lower.replace("_", " ")] = entry.canonical_name
                self._alias_map[canonical_lower.replace("-", " ")] = entry.canonical_name
                for alias in entry.aliases:
                    alias_lower = alias.lower()
                    self._alias_map[alias_lower.replace("_", " ")] = entry.canonical_name
                    self._alias_map[alias_lower.replace("-", " ")] = entry.canonical_name

        logger.debug(
            "SynonymMapper built with %d mappings from %d datasets",
            len(self._alias_map),
            len(self._datasets),
        )

    def map(self, label: str) -> str | None:
        """Return canonical name for a raw label, or None if not found."""
        normalized = label.lower().strip().replace("_", " ").replace("-", " ")
        canonical = self._alias_map.get(normalized)
        if canonical:
            logger.debug(
                "SynonymMapper | '%s' -> '%s'", label, canonical
            )
            return canonical
        logger.debug("SynonymMapper | '%s' -> no mapping found", label)
        return None

    def map_with_confidence(self, label: str) -> MappingResult:
        """Return mapping result with confidence."""
        canonical = self.map(label)
        if canonical is None:
            return MappingResult(canonical_name=None, confidence=0.0, source="none")
        return MappingResult(
            canonical_name=canonical,
            confidence=1.0,  # Direct synonym mapping is high confidence
            source="synonym",
        )

    def _find_entry(self, canonical_name: str) -> OntologyEntry | None:
        for dataset in self._datasets:
            entry = dataset.get_entry(canonical_name)
            if entry:
                return entry
        return None
