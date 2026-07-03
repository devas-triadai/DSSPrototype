"""Category Mapper.

Maps canonical concepts into hierarchical parent and child categories.
Respects the ontology taxonomy without introducing false specificity.
"""

import logging
from typing import Any

from backend.modules.knowledge.ontology.config import (
    CATEGORY_CONFIDENCE_DECAY,
    MAX_PARENT_DEPTH,
)
from backend.modules.knowledge.ontology.interfaces import CategoryMapperInterface
from backend.modules.knowledge.ontology.models import OntologyDataset, SemanticConcept

logger = logging.getLogger("dss.ontology.category_mapper")


class CategoryMapper(CategoryMapperInterface):
    """Maps concepts into hierarchical categories with confidence decay."""

    def __init__(self, datasets: list[OntologyDataset]) -> None:
        self._datasets = datasets

    def get_parents(self, canonical_name: str) -> list[str]:
        """Return parent categories ordered from closest to most general."""
        parents: list[str] = []
        current = canonical_name
        depth = 0
        visited: set[str] = set()

        while current and depth < MAX_PARENT_DEPTH:
            if current in visited:
                break
            visited.add(current)

            entry = self._find_entry(current)
            if entry is None or entry.parent_category is None:
                break

            parent = entry.parent_category
            if parent and parent not in visited:
                parents.append(parent)
                current = parent
            else:
                break
            depth += 1

        return parents

    def get_children(self, canonical_name: str) -> list[str]:
        """Return direct child categories."""
        entry = self._find_entry(canonical_name)
        if entry is None:
            return []
        return list(entry.child_categories)

    def get_expanded_concepts(
        self,
        canonical_name: str,
        base_confidence: float,
    ) -> list[SemanticConcept]:
        """Return all parent concepts with decayed confidence."""
        concepts: list[SemanticConcept] = []
        parents = self.get_parents(canonical_name)
        confidence = base_confidence

        for i, parent in enumerate(parents):
            confidence *= CATEGORY_CONFIDENCE_DECAY
            concepts.append(
                SemanticConcept(
                    canonical_name=parent,
                    confidence=round(confidence, 3),
                    source="category",
                    parent_concepts=parents[i + 1 :] if i + 1 < len(parents) else [],
                    child_concepts=[canonical_name] if i == 0 else [],
                )
            )
            logger.debug(
                "CategoryMapper | '%s' -> parent '%s' (confidence=%.3f)",
                canonical_name,
                parent,
                confidence,
            )

        return concepts

    def _find_entry(self, canonical_name: str) -> Any | None:
        for dataset in self._datasets:
            entry = dataset.get_entry(canonical_name)
            if entry:
                return entry
        return None
