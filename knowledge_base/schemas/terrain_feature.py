"""Canonical schema for terrain feature documents.

Represents geographic and terrain features for mobility,
visibility, and tactical analysis.
"""

from dataclasses import dataclass, field
from typing import Any

from knowledge_base.schemas.base import BaseDocument, DocumentType, _dataclass_to_dict


@dataclass(frozen=True)
class GeoLocation:
    """Geographic coordinates for a terrain feature."""

    latitude: float = 0.0
    longitude: float = 0.0


@dataclass(frozen=True)
class GeoBounds:
    """Axis-aligned bounding box in geographic coordinates."""

    min_lat: float = 0.0
    min_lon: float = 0.0
    max_lat: float = 0.0
    max_lon: float = 0.0


@dataclass(frozen=True)
class TerrainFeatureDocument(BaseDocument):
    """A terrain feature with geographic, physical, and tactical attributes.

    Supports point locations (``location``) and area features (``bounds``).
    At least one of ``terrain_type`` or ``name`` should be provided for
    query matching.
    """

    name: str = ""
    terrain_type: str = ""
    location: GeoLocation | None = None
    bounds: GeoBounds | None = None
    elevation: float | None = None
    region: str = ""
    country: str = ""
    features: list[str] = field(default_factory=list)
    road_network: list[str] = field(default_factory=list)
    water_bodies: list[str] = field(default_factory=list)
    vegetation: list[str] = field(default_factory=list)
    obstacles: list[str] = field(default_factory=list)
    slope: float | None = None
    soil_type: str = ""
    visibility: str = ""
    road_access: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_type", DocumentType.TERRAIN_FEATURE)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(_dataclass_to_dict(self))
        result.pop("document_type", None)
        result["document_type"] = DocumentType.TERRAIN_FEATURE.value
        return result
