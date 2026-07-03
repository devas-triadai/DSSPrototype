"""Category-based search using the category index."""

from typing import Any

from knowledge_base.indexer.category_index import CategoryIndex
from knowledge_base.search.base import SearchResult


def category_search(
    index: CategoryIndex,
    documents: dict[str, dict[str, Any]],
    query: str,
    limit: int = 10,
) -> SearchResult:
    """Search documents by category matching.

    Parameters
    ----------
    index:
        A populated ``CategoryIndex``.
    documents:
        Mapping of ``document_id`` to document dict.
    query:
        Category name or prefix.
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
        method="category",
    )
