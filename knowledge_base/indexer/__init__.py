"""Document indexers for the DSS Knowledge Base.

Indexers build in-memory lookup structures from loaded and validated
documents, enabling fast search by keyword, category, and attribute.
"""

from knowledge_base.indexer.attribute_index import AttributeIndex
from knowledge_base.indexer.base import Index, IndexEntry
from knowledge_base.indexer.category_index import CategoryIndex
from knowledge_base.indexer.keyword_index import KeywordIndex

__all__ = [
    "Index",
    "IndexEntry",
    "KeywordIndex",
    "CategoryIndex",
    "AttributeIndex",
]
