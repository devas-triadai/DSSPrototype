"""Friendly-force knowledge retriever implementation.

Provides a concrete ``RetrieverInterface`` backed by the
``knowledge_base`` infrastructure (loader, validator, indexer,
search).
"""

from knowledge_base.friendly.retriever import FriendlyKnowledgeRetriever

__all__ = ["FriendlyKnowledgeRetriever"]
