"""Knowledge document metadata — version tracking, source attribution, and integrity.

All knowledge documents include embedded metadata for provenance,
versioning, and tamper detection.
"""

from knowledge_base.metadata.models import (
    ChecksumInfo,
    DatasetInfo,
    DocumentMetadata,
    SourceInfo,
)

__all__ = [
    "DocumentMetadata",
    "SourceInfo",
    "ChecksumInfo",
    "DatasetInfo",
]
