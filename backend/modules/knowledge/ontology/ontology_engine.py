"""Ontology Engine.

Coordinates synonym mapping, category expansion, alias resolution, and
confidence adjustment into a single ontology processing pipeline.
"""

import json
import logging
from pathlib import Path

from backend.modules.knowledge.ontology.alias_mapper import AliasMapper
from backend.modules.knowledge.ontology.category_mapper import CategoryMapper
from backend.modules.knowledge.ontology.confidence_adjuster import ConfidenceAdjuster
from backend.modules.knowledge.ontology.config import DEFAULT_ONTOLOGY_FILES
from backend.modules.knowledge.ontology.interfaces import OntologyEngineInterface
from backend.modules.knowledge.ontology.models import (
    OntologyDataset,
    OntologyEntry,
    OntologyResult,
    SemanticConcept,
)
from backend.modules.knowledge.ontology.synonym_mapper import SynonymMapper

logger = logging.getLogger("dss.ontology.engine")


class OntologyEngine(OntologyEngineInterface):
    """Coordinates all ontology processing components.

    Dependency-injected: accepts pre-built components or constructs defaults.
    """

    def __init__(
        self,
        datasets: list[OntologyDataset] | None = None,
        synonym_mapper: SynonymMapper | None = None,
        category_mapper: CategoryMapper | None = None,
        alias_mapper: AliasMapper | None = None,
        confidence_adjuster: ConfidenceAdjuster | None = None,
        ontology_files: dict[str, Path] | None = None,
    ) -> None:
        self._ontology_files = ontology_files or DEFAULT_ONTOLOGY_FILES
        self._datasets = datasets or self._load_datasets()
        self._synonym_mapper = synonym_mapper or SynonymMapper(self._datasets)
        self._category_mapper = category_mapper or CategoryMapper(self._datasets)
        self._alias_mapper = alias_mapper or AliasMapper(self._datasets)
        self._confidence_adjuster = confidence_adjuster or ConfidenceAdjuster()
        self._version = self._compute_version()

        logger.info(
            "OntologyEngine initialized | version=%s | domains=%s | entries=%d",
            self._version,
            list(self._ontology_files.keys()),
            sum(len(d.entries) for d in self._datasets),
        )

    def _load_datasets(self) -> list[OntologyDataset]:
        """Load all ontology JSON files into memory."""
        datasets: list[OntologyDataset] = []
        for domain, path in self._ontology_files.items():
            if not path.exists():
                logger.warning("Ontology file not found: %s", path)
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                entries = {
                    e["canonical_name"]: OntologyEntry(
                        canonical_name=e["canonical_name"],
                        aliases=list(e.get("aliases", [])),
                        parent_category=e.get("parent_category"),
                        child_categories=list(e.get("child_categories", [])),
                        civilian_equivalents=list(e.get("civilian_equivalents", [])),
                        military_equivalents=list(e.get("military_equivalents", [])),
                        description=e.get("description", ""),
                        confidence_base=float(e.get("confidence_base", 0.5)),
                    )
                    for e in raw.get("entries", [])
                }
                datasets.append(
                    OntologyDataset(
                        version=raw.get("version", "unknown"),
                        schema_version=raw.get("schema_version", "unknown"),
                        domain=raw.get("domain", domain),
                        description=raw.get("description", ""),
                        entries=entries,
                    )
                )
                logger.info(
                    "OntologyEngine | Loaded %s domain | %d entries | version=%s",
                    domain,
                    len(entries),
                    raw.get("version", "unknown"),
                )
            except Exception as exc:
                logger.error("OntologyEngine | Failed to load %s: %s", domain, exc)
        return datasets

    def _compute_version(self) -> str:
        """Compute a composite version string from all loaded datasets."""
        versions = [f"{d.domain}:{d.version}" for d in self._datasets]
        return "|".join(versions) if versions else "empty"

    def get_version(self) -> str:
        return self._version

    def process(
        self,
        label: str,
        detector_confidence: float,
    ) -> OntologyResult:
        """Transform a raw detector label into a structured ontology result.

        Pipeline:
        1. Synonym mapping (raw -> canonical)
        2. Category expansion (canonical -> parents)
        3. Alias expansion (canonical -> military/civilian equivalents)
        4. Confidence adjustment for all concepts
        """
        logger.info(
            "OntologyEngine | Processing label='%s' detector_confidence=%.3f",
            label,
            detector_confidence,
        )

        # Step 1: Synonym mapping
        canonical = self._synonym_mapper.map(label)
        mapping_path: list[str] = [label]
        if canonical:
            mapping_path.append(canonical)
            logger.info(
                "OntologyEngine | Canonical mapping: '%s' -> '%s'",
                label,
                canonical,
            )
        else:
            logger.info(
                "OntologyEngine | No canonical mapping for '%s' — using original",
                label,
            )

        # Step 2: Build expanded concepts
        expanded: list[SemanticConcept] = []
        base_confidence = detector_confidence

        if canonical:
            entry = self._find_entry(canonical)
            if entry:
                base_confidence = min(detector_confidence, entry.confidence_base)

            # Direct concept
            direct_conf = self._confidence_adjuster.adjust(
                detector_confidence, base_confidence, 0
            )
            expanded.append(
                SemanticConcept(
                    canonical_name=canonical,
                    confidence=direct_conf,
                    source="direct",
                    parent_concepts=[],
                    child_concepts=[],
                )
            )

            # Category parents
            category_concepts = self._category_mapper.get_expanded_concepts(
                canonical, base_confidence
            )
            for concept in category_concepts:
                adjusted_conf = self._confidence_adjuster.adjust_concept_confidence(
                    detector_confidence, concept.confidence, concept.source
                )
                expanded.append(
                    SemanticConcept(
                        canonical_name=concept.canonical_name,
                        confidence=adjusted_conf,
                        source=concept.source,
                        parent_concepts=concept.parent_concepts,
                        child_concepts=concept.child_concepts,
                    )
                )
                mapping_path.append(concept.canonical_name)

            # Military and civilian equivalents
            equivalent_concepts = self._alias_mapper.get_equivalent_concepts(
                canonical, base_confidence
            )
            for concept in equivalent_concepts:
                adjusted_conf = self._confidence_adjuster.adjust_concept_confidence(
                    detector_confidence, concept.confidence, concept.source
                )
                expanded.append(
                    SemanticConcept(
                        canonical_name=concept.canonical_name,
                        confidence=adjusted_conf,
                        source=concept.source,
                        parent_concepts=[],
                        child_concepts=[],
                    )
                )

        # Step 3: Compute overall adjusted confidence
        adjusted_confidence = self._confidence_adjuster.adjust(
            detector_confidence, base_confidence, len(mapping_path) - 1
        )

        # Deduplicate by canonical_name, keeping highest confidence
        seen: dict[str, SemanticConcept] = {}
        for concept in expanded:
            if concept.canonical_name in seen:
                if concept.confidence > seen[concept.canonical_name].confidence:
                    seen[concept.canonical_name] = concept
            else:
                seen[concept.canonical_name] = concept

        deduplicated = sorted(
            seen.values(), key=lambda c: c.confidence, reverse=True
        )

        result = OntologyResult(
            original_label=label,
            canonical_concept=canonical,
            expanded_concepts=deduplicated,
            adjusted_confidence=adjusted_confidence,
            ontology_version=self._version,
            mapping_path=mapping_path,
        )

        logger.info(
            "OntologyEngine | Result: canonical='%s' concepts=%d adjusted_conf=%.3f",
            canonical,
            len(deduplicated),
            adjusted_confidence,
        )
        return result

    def expand_query(self, query: str) -> list[str]:
        """Expand a query into all related search terms.

        Always includes the original query first.
        """
        result = self.process(query, detector_confidence=1.0)
        terms = result.get_all_search_terms()
        logger.info(
            "OntologyEngine | Query expansion: '%s' -> %s",
            query,
            terms,
        )
        return terms

    def _find_entry(self, canonical_name: str) -> OntologyEntry | None:
        for dataset in self._datasets:
            entry = dataset.get_entry(canonical_name)
            if entry:
                return entry
        return None
