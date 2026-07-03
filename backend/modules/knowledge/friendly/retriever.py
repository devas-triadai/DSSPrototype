"""Knowledge retrieval for friendly-force assessment.

Provides a base ``NullRetriever`` that returns empty results and a
framework for implementing concrete retrievers backed by JSON files,
vector databases, PDF parsers, or remote APIs.
"""

import logging
from typing import Any

from backend.modules.knowledge.friendly.config import friendly_config
from backend.modules.knowledge.friendly.interfaces import (
    RetrievalResult,
    RetrieverInterface,
)

logger = logging.getLogger("dss.knowledge.friendly.retriever")


class NullRetriever(RetrieverInterface):
    """A retriever that always returns empty results.

    This is the default implementation.  Replace it with a concrete
    retriever via dependency injection once a knowledge source is
    available (vector DB, JSON file, API, etc.).
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or friendly_config

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
