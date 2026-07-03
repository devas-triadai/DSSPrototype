"""Keyword-based index — maps lowercase tokens to document IDs."""

import re
from collections import defaultdict
from typing import Any

from knowledge_base.indexer.base import Index, IndexEntry


class KeywordIndex(Index):
    """Inverted index mapping individual keywords to documents.

    Keywords are extracted from the ``name``, ``category``,
    ``terrain_type``, ``role``, and ``tags`` fields and tokenised
    on whitespace and non-alphanumeric boundaries.
    """

    def __init__(self) -> None:
        self._inverted: dict[str, dict[str, float]] = defaultdict(dict)
        self._doc_titles: dict[str, str] = {}

    def build(self, documents: list[dict[str, Any]]) -> None:
        self._inverted.clear()
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

            text_fields: list[str] = []
            for field in ("name", "category", "terrain_type", "role", "tags"):
                value = doc.get(field)
                if isinstance(value, str):
                    text_fields.append(value)
                elif isinstance(value, list):
                    text_fields.extend(str(v) for v in value)

            tokens = set()
            for text in text_fields:
                for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
                    if len(token) >= 2:
                        tokens.add(token)

            for token in tokens:
                if doc_id not in self._inverted[token]:
                    self._inverted[token][doc_id] = 1.0
                else:
                    self._inverted[token][doc_id] += 0.5

    def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[IndexEntry]:
        tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        if not tokens:
            return []

        scores: dict[str, float] = {}
        seen_docs: dict[str, set[str]] = {}
        for token in tokens:
            matches = self._inverted.get(token, {})
            for doc_id, score in matches.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + score
                seen_docs.setdefault(doc_id, set()).add(token)

        boost = 1.5
        for doc_id in list(scores.keys()):
            matched_tokens = seen_docs[doc_id]
            fraction = len(matched_tokens) / max(len(tokens), 1)
            scores[doc_id] *= 1.0 + (fraction * boost)

        sorted_docs = sorted(
            scores.items(),
            key=lambda x: (-x[1], x[0]),
        )

        return [
            IndexEntry(
                document_id=doc_id,
                key=token,
                score=score,
            )
            for doc_id, score in sorted_docs[:limit]
            for token in seen_docs.get(doc_id, set())
        ]
