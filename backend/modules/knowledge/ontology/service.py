"""Ontology Service.

Service-level wrapper for the ontology engine.  Provides a clean
injection point for the knowledge modules.
"""

import logging

from backend.modules.knowledge.ontology.interfaces import (
    OntologyEngineInterface,
    OntologyServiceInterface,
)
from backend.modules.knowledge.ontology.models import OntologyResult
from backend.modules.knowledge.ontology.ontology_engine import OntologyEngine

logger = logging.getLogger("dss.ontology.service")


class OntologyService(OntologyServiceInterface):
    """Service wrapper for ontology processing.

    Accepts an optional pre-built engine via dependency injection.
    If none is provided, constructs a default engine with all standard
    ontology files.
    """

    def __init__(self, engine: OntologyEngineInterface | None = None) -> None:
        self._engine = engine or OntologyEngine()
        logger.info(
            "OntologyService initialized | engine_version=%s",
            self._engine.get_version(),
        )

    def process(
        self,
        label: str,
        detector_confidence: float,
    ) -> OntologyResult:
        """Process a detector label through the ontology pipeline."""
        return self._engine.process(label, detector_confidence)

    def expand_query(self, query: str) -> list[str]:
        """Expand a query string into all related ontology search terms."""
        return self._engine.expand_query(query)

    def get_version(self) -> str:
        """Return the loaded ontology version."""
        return self._engine.get_version()
