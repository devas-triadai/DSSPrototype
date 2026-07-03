"""Abstract searcher interface and shared result type."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SearchResult:
    """Result of a knowledge base search.

    Attributes
    ----------
    query:
        The original search query string.
    documents:
        Matching document dictionaries.
    count:
        Number of results returned.
    total:
        Total number of matches (may exceed ``count`` if limited).
    method:
        Description of the search method used.
    """

    query: str = ""
    documents: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0
    total: int = 0
    method: str = ""


class Searcher(Protocol):
    """Interface for all search implementations."""

    def search(self, query: str, limit: int = 10) -> SearchResult: ...
