"""Abstract index interface and shared types."""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class IndexEntry:
    """A single indexed entry referencing a document.

    Attributes
    ----------
    document_id:
        Unique identifier for the document.
    key:
        The index key this entry is stored under.
    score:
        Relevance score for ranking (higher = more relevant).
    """

    document_id: str
    key: str = ""
    score: float = 1.0


class Index(Protocol):
    """Interface for all document indexes.

    An index builds its internal structure from a list of document
    dictionaries, and provides a ``search`` method.
    """

    def build(self, documents: list[dict[str, Any]]) -> None: ...

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[IndexEntry]: ...  # pragma: no cover
