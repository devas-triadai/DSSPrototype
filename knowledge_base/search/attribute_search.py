"""Attribute-based search using the attribute index."""

from typing import Any

from knowledge_base.indexer.attribute_index import AttributeIndex
from knowledge_base.search.base import SearchResult


def attribute_search(
    index: AttributeIndex,
    documents: dict[str, dict[str, Any]],
    query: str,
    limit: int = 10,
) -> SearchResult:
    """Search documents by attribute value matching.

    Parameters
    ----------
    index:
        A populated ``AttributeIndex``.
    documents:
        Mapping of ``document_id`` to document dict.
    query:
        Attribute value or prefix (e.g. "India", "BAE").
    limit:
        Maximum results to return.

    Returns
    -------
    SearchResult
    """
    entries = index.search(query, limit=limit)

    matched: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        doc = documents.get(entry.document_id)
        if doc and entry.document_id not in seen:
            matched.append(doc)
            seen.add(entry.document_id)

    return SearchResult(
        query=query,
        documents=matched,
        count=len(matched),
        total=len(matched),
        method="attribute",
    )
