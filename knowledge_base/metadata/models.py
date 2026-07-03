"""Data models for knowledge document metadata.

Every knowledge document carries embedded metadata that records its
provenance, version history, licensing, and integrity checksum.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class SourceInfo:
    """Attribution and provenance for a knowledge document."""

    name: str
    type: str
    reference: str | None = None
    classification: str = "unclassified"


@dataclass(frozen=True)
class ChecksumInfo:
    """Integrity verification data for a knowledge document."""

    algorithm: str = "sha256"
    value: str = ""
    verified_at: str = ""


@dataclass(frozen=True)
class DocumentMetadata:
    """Embedded metadata carried by every knowledge document.

    Provides version tracking, source attribution, licensing,
    and integrity verification.  New fields may be added but
    existing fields must never be removed or renamed.
    """

    document_id: str
    version: str = "1.0.0"
    title: str = ""
    description: str = ""
    source: SourceInfo | None = None
    license: str = "Proprietary — demonstration use only"
    author: str = "DSSPrototype"
    created_at: str = ""
    updated_at: str = ""
    checksum: ChecksumInfo | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            )
        if not self.updated_at:
            object.__setattr__(
                self,
                "updated_at",
                datetime.now(timezone.utc).isoformat(),
            )


@dataclass(frozen=True)
class DatasetInfo:
    """Aggregate metadata for an entire dataset file."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    source: SourceInfo | None = None
    license: str = "Proprietary — demonstration use only"
    document_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    checksum: ChecksumInfo | None = None
