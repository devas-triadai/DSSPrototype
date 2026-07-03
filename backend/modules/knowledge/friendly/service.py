"""Public entry point for the Friendly Knowledge module.

Coordinates the full friendly-force assessment pipeline while
containing zero retrieval, evidence, or scoring logic.
"""

import logging
from typing import Any

from backend.contracts.interfaces.friendly import FriendlyModule
from backend.contracts.models.analysis import FriendlyAnalysis
from backend.contracts.models.detection import DetectionResult
from backend.modules.knowledge.friendly.config import friendly_config
from backend.modules.knowledge.friendly.interfaces import (
    KnowledgeEngineInterface,
    RetrieverInterface,
)
from backend.modules.knowledge.friendly.knowledge_engine import KnowledgeEngine
from knowledge_base.friendly.retriever import FriendlyKnowledgeRetriever

logger = logging.getLogger("dss.knowledge.friendly.service")


class FriendlyKnowledgeService(FriendlyModule):
    """Orchestrates the end-to-end friendly knowledge pipeline.

    Pipeline steps:
        1. Delegate to ``KnowledgeEngine`` for reasoning.
        2. Return a strongly typed ``FriendlyAnalysis``.

    Dependencies are injected via the constructor; sensible defaults
    are provided for every component.
    """

    def __init__(
        self,
        knowledge_engine: KnowledgeEngineInterface | None = None,
        retriever: RetrieverInterface | None = None,
        config: Any | None = None,
        ontology_service: Any | None = None,
    ) -> None:
        self._knowledge_engine = knowledge_engine or KnowledgeEngine()
        self._ontology_service = ontology_service
        self._retriever = retriever or FriendlyKnowledgeRetriever(
            ontology_service=ontology_service
        )
        self._config = config or friendly_config

    async def analyze_friendly(self, detection: DetectionResult) -> FriendlyAnalysis:
        """Run the full friendly-knowledge pipeline on a detection result.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        FriendlyAnalysis
            Strongly typed friendly-force assessment.
        """
        logger.info(
            "Analyzing detection %s (%d objects) (ontology=%s)",
            detection.image_id,
            len(detection.objects),
            "enabled" if self._ontology_service else "disabled",
        )
        return await self._knowledge_engine.analyze(detection, self._retriever)
