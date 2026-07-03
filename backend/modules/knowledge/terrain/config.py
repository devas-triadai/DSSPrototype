"""Terrain Knowledge module configuration.

All values are overridable via environment variables prefixed with ``TERRAIN_``.
"""

from pydantic_settings import BaseSettings


class TerrainConfig(BaseSettings):
    """Configuration for the Terrain Knowledge pipeline.

    Controls retrieval, feature extraction, mobility analysis,
    visibility analysis, and future GIS / DEM / map-cache settings.
    """

    model_config = {"env_prefix": "TERRAIN_"}

    # Retrieval
    retriever_type: str = "null"
    retriever_endpoint: str = ""
    retriever_timeout_seconds: float = 30.0
    max_terrain_items: int = 20

    # Features
    max_features: int = 20

    # Mobility
    default_road_access: bool = False
    mobility_rating_high_threshold: float = 0.7
    mobility_rating_medium_threshold: float = 0.4

    # Visibility
    default_visibility: str = "unknown"

    # Confidence
    default_confidence: float = 0.5
    confidence_weight_detection: float = 0.3
    confidence_weight_terrain: float = 0.4
    confidence_weight_evidence: float = 0.3

    # Future — GIS
    gis_endpoint: str = ""
    gis_layer_name: str = ""
    gis_timeout_seconds: float = 30.0

    # Future — DEM / Elevation
    dem_endpoint: str = ""
    dem_grid_size: int = 30

    # Future — Map Cache
    cache_ttl_seconds: int = 600
    cache_max_size: int = 500

    # Future — Satellite
    satellite_tile_endpoint: str = ""
    satellite_tile_format: str = "tiff"


terrain_config = TerrainConfig()
