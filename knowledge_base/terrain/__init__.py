"""Terrain knowledge retriever implementation.

Provides a concrete ``TerrainRetrieverInterface`` backed by the
``knowledge_base`` infrastructure (loader, validator, indexer,
search).
"""

from knowledge_base.terrain.retriever import TerrainKnowledgeRetriever

__all__ = ["TerrainKnowledgeRetriever"]
