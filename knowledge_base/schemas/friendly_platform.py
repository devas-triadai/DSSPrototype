"""Canonical schema for friendly military platform documents.

Represents known friendly-force equipment, units, and platforms
for match-making against computer vision detections.
"""

from dataclasses import dataclass, field
from typing import Any

from knowledge_base.schemas.base import BaseDocument, DocumentType, _dataclass_to_dict


@dataclass(frozen=True)
class FriendlyPlatformDocument(BaseDocument):
    """A known friendly military platform with identifying characteristics.

    Fields are intentionally sparse to support heterogeneous data
    sources.  Only ``name``, ``category``, and ``platform_type`` are
    required at the schema level; all other fields are optional.
    """

    name: str = ""
    category: str = ""
    platform_type: str = ""
    country: str = ""
    manufacturer: str = ""
    role: str = ""
    characteristics: list[str] = field(default_factory=list)
    equipment: list[str] = field(default_factory=list)
    markings: list[str] = field(default_factory=list)
    weight_tonnes: float | None = None
    crew: int | None = None
    speed_kmh: float | None = None
    armament: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_type", DocumentType.FRIENDLY_PLATFORM)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(_dataclass_to_dict(self))
        result.pop("document_type", None)
        result["document_type"] = DocumentType.FRIENDLY_PLATFORM.value
        return result
