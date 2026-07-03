"""Abstract loader interface and shared result type."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LoaderResult:
    """Result of loading a dataset file.

    Attributes
    ----------
    documents:
        List of document dictionaries parsed from the file.
    source:
        Original file path or identifier.
    count:
        Number of documents loaded.
    errors:
        Any errors encountered during loading (non-fatal).
    """

    documents: list[dict[str, Any]]
    source: str = ""
    count: int = 0
    errors: list[str] = field(default_factory=list)


class Loader(Protocol):
    """Interface for all data loaders.

    Every loader reads a file from a given path and returns a
    ``LoaderResult`` containing parsed document dictionaries.
    """

    def load(self, path: str) -> LoaderResult: ...  # pragma: no cover
