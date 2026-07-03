"""Public entry point for the Enemy Knowledge module.

Coordinates the full enemy-force assessment pipeline while
containing zero retrieval, evidence, or scoring logic.
"""

import logging
from typing import Any

from backend.contracts.interfaces.enemy import EnemyModule
from backend.contracts.models.analysis import EnemyAnalysis
from backend.contracts.models.detection import DetectionResult
from backend.modules.knowledge.enemy.config import enemy_config
from backend.modules.knowledge.enemy.interfaces import (
    KnowledgeEngineInterface,
    RetrieverInterface,
)
from backend.modules.knowledge.enemy.knowledge_engine import KnowledgeEngine
from knowledge_base.enemy.retriever import EnemyKnowledgeRetriever

logger = logging.getLogger("dss.knowledge.enemy.service")


class EnemyKnowledgeService(EnemyModule):
    """Orchestrates the end-to-end enemy knowledge pipeline.

    Pipeline steps:
        1. Delegate to ``KnowledgeEngine`` for reasoning.
        2. Return a strongly typed ``EnemyAnalysis``.

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
        self._retriever = retriever or EnemyKnowledgeRetriever(
            ontology_service=ontology_service
        )
        self._config = config or enemy_config

    async def analyze_enemy(self, detection: DetectionResult) -> EnemyAnalysis:
        """Run the full enemy-knowledge pipeline on a detection result.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        EnemyAnalysis
            Strongly typed enemy-force assessment.
        """
        logger.info(
            "Analyzing detection %s (%d objects) (ontology=%s)",
            detection.image_id,
            len(detection.objects),
            "enabled" if self._ontology_service else "disabled",
        )
        return await self._knowledge_engine.analyze(detection, self._retriever)
