"""Verify knowledge_base imports work from project root."""
import sys
sys.path.insert(0, ".")

from knowledge_base.friendly.retriever import FriendlyKnowledgeRetriever
from knowledge_base.enemy.retriever import EnemyKnowledgeRetriever
from knowledge_base.terrain.retriever import TerrainKnowledgeRetriever

print("knowledge_base imports OK")

r = FriendlyKnowledgeRetriever()
print(f"FriendlyKnowledgeRetriever: {type(r).__name__}")
print(f"  has retrieve: {hasattr(r, 'retrieve')}")

# Verify it can load and search
import asyncio
async def test():
    result = await r.retrieve("Arjun")
    print(f"  search 'Arjun': {len(result.items)} items, {result.query_time_ms:.1f}ms")
    for item in result.items[:2]:
        print(f"    - {item.unit_name} ({item.source})")

asyncio.run(test())
