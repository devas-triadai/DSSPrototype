"""Ontology module.

Military Object Ontology and Label Normalization Layer.

This module sits between Computer Vision and the Knowledge Retrievers,
transforming raw detector labels into standardized semantic concepts
that can be searched more effectively in the knowledge base.

Components:
- OntologyEngine: coordinates all processing
- SynonymMapper: maps equivalent labels to canonical forms
- CategoryMapper: maps concepts into hierarchical categories
- AliasMapper: resolves alternate names and equivalents
- ConfidenceAdjuster: adjusts confidence based on mapping quality
- OntologyService: service-level wrapper for DI

Usage:
    from backend.modules.knowledge.ontology import OntologyService

    service = OntologyService()
    result = service.process("truck", detector_confidence=0.85)
    # result.canonical_concept == "truck"
    # result.expanded_concepts includes "wheeled_vehicle", "ground_vehicle", etc.
"""

from backend.modules.knowledge.ontology.interfaces import (
    AliasMapperInterface,
    CategoryMapperInterface,
    ConfidenceAdjusterInterface,
    OntologyEngineInterface,
    OntologyServiceInterface,
    SynonymMapperInterface,
)
from backend.modules.knowledge.ontology.models import (
    OntologyDataset,
    OntologyEntry,
    OntologyResult,
    SemanticConcept,
)
from backend.modules.knowledge.ontology.ontology_engine import OntologyEngine
from backend.modules.knowledge.ontology.service import OntologyService

__all__ = [
    "OntologyService",
    "OntologyEngine",
    "OntologyResult",
    "SemanticConcept",
    "OntologyEntry",
    "OntologyDataset",
    "OntologyEngineInterface",
    "OntologyServiceInterface",
    "SynonymMapperInterface",
    "CategoryMapperInterface",
    "AliasMapperInterface",
    "ConfidenceAdjusterInterface",
]
