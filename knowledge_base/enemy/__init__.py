"""Enemy-force knowledge retriever implementation.

Provides a concrete ``RetrieverInterface`` backed by the
``knowledge_base`` infrastructure (loader, validator, indexer,
search).
"""

from knowledge_base.enemy.retriever import EnemyKnowledgeRetriever

__all__ = ["EnemyKnowledgeRetriever"]
