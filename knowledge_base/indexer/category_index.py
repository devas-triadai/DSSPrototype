"""Category-based index — groups documents by their ``category`` field."""

from collections import defaultdict
from typing import Any

from knowledge_base.indexer.base import Index, IndexEntry


class CategoryIndex(Index):
    """Index that groups documents by their ``category`` field.

    Supports exact category lookups and prefix-based partial matches.
    """

    def __init__(self) -> None:
        self._by_category: dict[str, list[str]] = defaultdict(list)
        self._doc_titles: dict[str, str] = {}

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._by_category.clear()
        self._doc_titles.clear()

        for doc in documents:
            doc_id = (
                doc.get("document_id")
                or doc.get("metadata", {}).get("document_id")
                or ""
            )
            if not doc_id:
                continue

            title = doc.get("name") or doc.get("terrain_type") or ""
            self._doc_titles[doc_id] = str(title)

            category = doc.get("category", "")
            if isinstance(category, str) and category:
                self._by_category[category.lower()].append(doc_id)

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[IndexEntry]:
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        results: list[IndexEntry] = []
        seen: set[str] = set()

        for cat_name, doc_ids in self._by_category.items():
            if query_lower in cat_name:
                for doc_id in doc_ids:
                    if doc_id not in seen:
                        results.append(
                            IndexEntry(
                                document_id=doc_id,
                                key=cat_name,
                                score=1.0 if cat_name == query_lower else 0.5,
                            )
                        )
                        seen.add(doc_id)
                        if len(results) >= limit:
                            return results

        return results
