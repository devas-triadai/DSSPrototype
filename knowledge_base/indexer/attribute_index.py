"""Attribute-based index — maps specific field values to documents."""

from collections import defaultdict
from typing import Any

from knowledge_base.indexer.base import Index, IndexEntry


class AttributeIndex(Index):
    """Index that maps field-specific attribute values to documents.

    Useful for queries like "tanks with weight > 40 tonnes" or
    "vehicles manufactured by BAE Systems".
    """

    def __init__(self) -> None:
        self._by_attribute: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._doc_titles: dict[str, str] = {}

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._by_attribute.clear()
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

            for field_name in ("country", "manufacturer", "terrain_type", "platform_type"):
                value = doc.get(field_name)
                if isinstance(value, str) and value:
                    self._by_attribute[field_name][value.lower()].append(doc_id)

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

        for field_name, values in self._by_attribute.items():
            for attr_value, doc_ids in values.items():
                if query_lower in attr_value:
                    for doc_id in doc_ids:
                        if doc_id not in seen:
                            results.append(
                                IndexEntry(
                                    document_id=doc_id,
                                    key=f"{field_name}:{attr_value}",
                                    score=1.0,
                                )
                            )
                            seen.add(doc_id)
                            if len(results) >= limit:
                                return results

        return results
