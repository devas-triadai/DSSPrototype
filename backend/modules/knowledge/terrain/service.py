"""Public entry point for the Terrain Knowledge module.

Coordinates the full terrain-assessment pipeline while
containing zero retrieval, feature extraction, mobility,
visibility, or scoring logic.
"""

import logging
from typing import Any

from backend.contracts.interfaces.terrain import TerrainModule
from backend.contracts.models.analysis import TerrainAnalysis
from backend.contracts.models.detection import DetectionResult
from backend.modules.knowledge.terrain.config import terrain_config
from backend.modules.knowledge.terrain.interfaces import (
    RetrieverInterface,
    TerrainEngineInterface,
)
from backend.modules.knowledge.terrain.terrain_engine import TerrainEngine
from knowledge_base.terrain.retriever import TerrainKnowledgeRetriever

logger = logging.getLogger("dss.knowledge.terrain.service")


class TerrainKnowledgeService(TerrainModule):
    """Orchestrates the end-to-end terrain knowledge pipeline.

    Pipeline steps:
        1. Delegate to ``TerrainEngine`` for reasoning.
        2. Return a strongly typed ``TerrainAnalysis``.

    Dependencies are injected via the constructor; sensible defaults
    are provided for every component.
    """

    def __init__(
        self,
        terrain_engine: TerrainEngineInterface | None = None,
        retriever: RetrieverInterface | None = None,
        config: Any | None = None,
        ontology_service: Any | None = None,
    ) -> None:
        self._terrain_engine = terrain_engine or TerrainEngine()
        self._ontology_service = ontology_service
        self._retriever = retriever or TerrainKnowledgeRetriever(
            ontology_service=ontology_service
        )
        self._config = config or terrain_config

    async def analyze_terrain(self, detection: DetectionResult) -> TerrainAnalysis:
        """Run the full terrain-knowledge pipeline on a detection result.

        Parameters
        ----------
        detection:
            The output of a vision-detection pass.

        Returns
        -------
        TerrainAnalysis
            Strongly typed terrain assessment.
        """
        logger.info(
            "Analyzing terrain for detection %s (%d objects) (ontology=%s)",
            detection.image_id,
            len(detection.objects),
            "enabled" if self._ontology_service else "disabled",
        )
        return await self._terrain_engine.analyze(detection, self._retriever)
