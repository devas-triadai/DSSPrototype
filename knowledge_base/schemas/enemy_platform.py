"""Canonical schema for enemy military platform documents.

Represents known hostile-force equipment, units, and platforms
for threat assessment and match-making against computer vision
detections.
"""

from dataclasses import dataclass, field
from typing import Any

from knowledge_base.schemas.base import BaseDocument, DocumentType, _dataclass_to_dict


@dataclass(frozen=True)
class EnemyPlatformDocument(BaseDocument):
    """A known enemy military platform with identifying and threat characteristics.

    Includes additional fields for threat assessment: capabilities,
    threat indicators, and tactical role.
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
    capabilities: list[str] = field(default_factory=list)
    threat_indicators: list[str] = field(default_factory=list)
    tactical_role: str = ""
    weight_tonnes: float | None = None
    crew: int | None = None
    speed_kmh: float | None = None
    armament: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    countermeasures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_type", DocumentType.ENEMY_PLATFORM)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(_dataclass_to_dict(self))
        result.pop("document_type", None)
        result["document_type"] = DocumentType.ENEMY_PLATFORM.value
        return result
