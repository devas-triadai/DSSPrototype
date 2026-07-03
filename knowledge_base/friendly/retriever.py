"""Concrete retriever for friendly-force knowledge documents.

Implements RetrieverInterface from the existing
backend/modules/knowledge/friendly/interfaces.py contract.
"""

import logging
import time
from pathlib import Path
from typing import Any

from backend.modules.knowledge.friendly.interfaces import (
    KnowledgeItem,
    RetrievalResult,
    RetrieverInterface,
)
from knowledge_base.indexer.attribute_index import AttributeIndex
from knowledge_base.indexer.category_index import CategoryIndex
from knowledge_base.indexer.keyword_index import KeywordIndex
from knowledge_base.loader.json_loader import JsonLoader
from knowledge_base.search.attribute_search import attribute_search
from knowledge_base.search.category_search import category_search
from knowledge_base.search.keyword_search import keyword_search
from knowledge_base.validator.duplicate_validator import DuplicateValidator
from knowledge_base.validator.field_validator import FieldValidator
from knowledge_base.validator.schema_validator import SchemaValidator

logger = logging.getLogger("dss.knowledge_base.friendly.retriever")


def _get_default_dataset_path() -> str:
    """Return the canonical dataset path from the registry.

    Uses a lazy import to avoid a circular dependency with
    knowledge_base.registry (the registry imports this retriever).
    """
    from knowledge_base.registry import get_dataset_registry_entry

    entry = get_dataset_registry_entry("friendly_platforms")
    if entry is not None:
        return entry.default_dataset_path
    # Fallback — should never happen if the registry is initialised correctly.
    here = Path(__file__).resolve().parent.parent / "datasets"
    return str(here / "friendly_platforms.json")


class FriendlyKnowledgeRetriever(RetrieverInterface):
    """Retrieves friendly-force platform knowledge from the knowledge base."""

    def __init__(
        self,
        dataset_path: str | None = None,
        ontology_service: Any | None = None,
    ) -> None:
        self._dataset_path = dataset_path or _get_default_dataset_path()
        self._ontology_service = ontology_service
        self._documents: list[dict[str, Any]] = []
        self._docs_by_id: dict[str, dict[str, Any]] = {}
        self._keyword_index = KeywordIndex()
        self._category_index = CategoryIndex()
        self._attribute_index = AttributeIndex()
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        logger.info(
            "FriendlyKnowledgeRetriever | Loading dataset: %s",
            Path(self._dataset_path).name,
        )
        loader = JsonLoader()
        result = loader.load(self._dataset_path)
        for v in [SchemaValidator(), FieldValidator(), DuplicateValidator()]:
            vresult = v.validate(result.documents)
            if not vresult.valid:
                raise RuntimeError(
                    f"Friendly dataset validation failed: {vresult.errors}"
                )
        self._documents = result.documents
        self._docs_by_id = {
            doc.get("document_id") or str(i): doc
            for i, doc in enumerate(result.documents)
        }
        self._keyword_index.build(result.documents)
        self._category_index.build(result.documents)
        self._attribute_index.build(result.documents)
        self._loaded = True
        logger.info(
            "FriendlyKnowledgeRetriever | Loaded %d records | Index created | "
            "Search engine initialized",
            len(result.documents),
        )

    def _doc_to_knowledge_item(self, doc: dict[str, Any]) -> KnowledgeItem:
        return KnowledgeItem(
            source=doc.get("source", "knowledge_base"),
            unit_id=doc.get("document_id"),
            unit_name=doc.get("name"),
            equipment=list(doc.get("equipment", [])),
            characteristics=list(doc.get("characteristics", [])),
            markings=list(doc.get("markings", [])),
            confidence=0.7,
            metadata={
                k: v
                for k, v in doc.items()
                if k
                not in (
                    "document_type",
                    "document_id",
                    "name",
                    "equipment",
                    "characteristics",
                    "markings",
                    "metadata",
                )
            },
        )

    def _search_with_expansion(
        self,
        query: str,
        context: dict[str, Any] | None,
    ) -> tuple[list[dict[str, Any]], str, int]:
        """Search with ontology expansion.

        Returns (documents, search_type, total_results).
        """
        # Build expanded query set
        queries = [query]
        if self._ontology_service is not None:
            expanded = self._ontology_service.expand_query(query)
            for term in expanded:
                if term not in queries:
                    queries.append(term)
            logger.info(
                "FriendlyKnowledgeRetriever | Ontology expanded '%s' -> %s",
                query,
                queries,
            )

        # Search each term and collect results
        all_docs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        search_type = "attribute"  # default fallback

        for q in queries:
            # Keyword search for this term
            kw = keyword_search(self._keyword_index, self._docs_by_id, q)
            if kw.count > 0:
                search_type = "keyword"
                for doc in kw.documents:
                    doc_id = doc.get("document_id", str(id(doc)))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                continue  # found results, skip to next term

            # Category search for this term
            cat = category_search(self._category_index, self._docs_by_id, q)
            if cat.count > 0:
                search_type = "category"
                for doc in cat.documents:
                    doc_id = doc.get("document_id", str(id(doc)))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)
                continue

            # Attribute search for this term
            attr = attribute_search(self._attribute_index, self._docs_by_id, q)
            if attr.count > 0:
                search_type = "attribute"
                for doc in attr.documents:
                    doc_id = doc.get("document_id", str(id(doc)))
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_docs.append(doc)

        return all_docs, search_type, len(all_docs)

    async def retrieve(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        self._ensure_loaded()
        total_start = time.perf_counter()

        logger.info(
            "FriendlyKnowledgeRetriever | Searching Friendly KB | Query: %s",
            query,
        )

        docs, search_type, total_results = self._search_with_expansion(query, context)

        items = [self._doc_to_knowledge_item(doc) for doc in docs]
        elapsed_ms = (time.perf_counter() - total_start) * 1000.0

        logger.info(
            "FriendlyKnowledgeRetriever | Final knowledge items: %d | "
            "Search type: %s | Search time: %.1f ms",
            len(items),
            search_type,
            elapsed_ms,
        )
        return RetrievalResult(items=items, query_time_ms=elapsed_ms)
