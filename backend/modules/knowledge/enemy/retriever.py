"""Intelligence retrieval for enemy-force assessment.

Provides a base ``NullRetriever`` that returns empty results and a
framework for implementing concrete retrievers backed by threat
libraries, equipment catalogues, vector databases, or remote APIs.
"""

import logging
from typing import Any

from backend.modules.knowledge.enemy.config import enemy_config
from backend.modules.knowledge.enemy.interfaces import (
    RetrievalResult,
    RetrieverInterface,
)

logger = logging.getLogger("dss.knowledge.enemy.retriever")


class NullRetriever(RetrieverInterface):
    """A retriever that always returns empty results.

    This is the default implementation.  Replace it with a concrete
    retriever via dependency injection once an intelligence source is
    available (vector DB, threat library, API, etc.).
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or enemy_config

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
