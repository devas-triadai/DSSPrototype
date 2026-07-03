"""Base document schema and shared types for all knowledge documents."""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from knowledge_base.metadata.models import DocumentMetadata, SourceInfo


class DocumentType(str, Enum):
    """Enumeration of all canonical knowledge document types."""

    FRIENDLY_PLATFORM = "friendly_platform"
    ENEMY_PLATFORM = "enemy_platform"
    TERRAIN_FEATURE = "terrain_feature"


@dataclass(frozen=True)
class BaseDocument:
    """Base class for all knowledge documents.

    Every document carries embedded ``DocumentMetadata`` and a
    ``document_type`` discriminator.
    """

    document_type: DocumentType
    metadata: DocumentMetadata | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the document to a JSON-compatible dictionary.

        Override in subclasses to add document-specific fields.
        """
        result: dict[str, Any] = {
            "document_type": self.document_type.value,
        }
        if self.metadata is not None:
            result["metadata"] = _dataclass_to_dict(self.metadata)
        return result


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    """Recursively convert a frozen dataclass to a plain dict."""
    result: dict[str, Any] = {}
    for field_name in getattr(obj, "__dataclass_fields__", {}):
        value = getattr(obj, field_name)
        if value is None:
            continue
        if hasattr(value, "__dataclass_fields__"):
            result[field_name] = _dataclass_to_dict(value)
        elif isinstance(value, list):
            result[field_name] = [
                _dataclass_to_dict(v) if hasattr(v, "__dataclass_fields__") else v
                for v in value
            ]
        elif isinstance(value, Enum):
            result[field_name] = value.value
        else:
            result[field_name] = value
    return result


def make_metadata(
    document_id: str,
    title: str = "",
    description: str = "",
    version: str = "1.0.0",
    source_name: str = "DSSPrototype Knowledge Base",
    source_type: str = "curated",
    license_text: str = "Proprietary — demonstration use only",
    tags: list[str] | None = None,
) -> DocumentMetadata:
    """Convenience factory for creating ``DocumentMetadata`` instances."""
    return DocumentMetadata(
        document_id=document_id,
        version=version,
        title=title,
        description=description,
        source=SourceInfo(name=source_name, type=source_type),
        license=license_text,
        tags=tags or [],
    )
