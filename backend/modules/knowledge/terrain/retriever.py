"""Terrain information retrieval for terrain assessment.

Provides a base ``NullRetriever`` that returns empty results and a
framework for implementing concrete retrievers backed by GIS services,
DEM rasters, satellite tile servers, or offline map databases.
"""

import logging
from typing import Any

from backend.modules.knowledge.terrain.config import terrain_config
from backend.modules.knowledge.terrain.interfaces import (
    RetrievalResult,
    RetrieverInterface,
)

logger = logging.getLogger("dss.knowledge.terrain.retriever")


class NullRetriever(RetrieverInterface):
    """A retriever that always returns empty results.

    This is the default implementation.  Replace it with a concrete
    retriever via dependency injection once a terrain source is
    available (GIS service, DEM, satellite API, etc.).
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or terrain_config

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Return an empty retrieval result.

        Logs the query for observability.  No actual retrieval is
        performed.
        """
        logger.debug("NullRetriever received query: '%s' (context=%s)", query, context)
        return RetrievalResult(items=[], query_time_ms=0.0)
