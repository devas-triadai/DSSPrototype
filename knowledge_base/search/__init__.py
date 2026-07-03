"""Search interfaces for the DSS Knowledge Base.

Provides high-level search functions that compose multiple index
lookups into unified results.
"""

from knowledge_base.search.attribute_search import attribute_search
from knowledge_base.search.base import Searcher, SearchResult
from knowledge_base.search.category_search import category_search
from knowledge_base.search.keyword_search import keyword_search

__all__ = [
    "Searcher",
    "SearchResult",
    "keyword_search",
    "category_search",
    "attribute_search",
]
