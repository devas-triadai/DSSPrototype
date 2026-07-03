"""Terrain feature extraction from retrieved terrain data.

Translates raw terrain information into structured, explainable
terrain features.  No confidence, mobility, or visibility
calculations are performed.
"""

import logging
from typing import Any

from backend.modules.knowledge.terrain.config import terrain_config
from backend.modules.knowledge.terrain.interfaces import (
    TerrainData,
    TerrainFeature,
    TerrainFeatureBuilderInterface,
)

logger = logging.getLogger("dss.knowledge.terrain.terrain_feature_builder")


class TerrainFeatureBuilder(TerrainFeatureBuilderInterface):
    """Builds terrain features by extracting structured information from raw data.

    Feature types produced:

    * ``vegetation``      — forests, fields, grass, scrub.
    * ``water``           — rivers, lakes, ponds, streams.
    * ``road``            — paved roads, dirt tracks, highways.
    * ``obstacle``        — walls, ditches, berms, rubble.
    * ``elevation``       — hills, valleys, ridges, slope.
    * ``urban``           — buildings, settlements, structures.
    * ``bridge``          — bridges, culverts, crossings.
    * ``soil``            — soil type, ground composition.
    * ``weather``         — rain, snow, fog, wind.
    """

    def __init__(self, config: Any | None = None) -> None:
        self._config = config or terrain_config

    def build_features(
        self,
        data: list[TerrainData],
    ) -> list[TerrainFeature]:
        """Create terrain features from a list of retrieved terrain data items.

        Each data item is scanned for known feature categories.  The
        method is stateless — all features are derived from the input.
        """
        features: list[TerrainFeature] = []
        max_items = self._config.max_features

        for item in data:
            if len(features) >= max_items:
                break

            features.extend(self._extract_vegetation(item))
            features.extend(self._extract_water(item))
            features.extend(self._extract_roads(item))
            features.extend(self._extract_obstacles(item))
            features.extend(self._extract_elevation(item))
            features.extend(self._extract_urban(item))
            features.extend(self._extract_bridges(item))
            features.extend(self._extract_soil(item))
            features.extend(self._extract_weather(item))

        return features

    # ------------------------------------------------------------------
    # Private extractors
    # ------------------------------------------------------------------

    def _extract_vegetation(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        for veg in item.vegetation:
            result.append(TerrainFeature(
                feature_type="vegetation",
                description=f"Vegetation: {veg}",
                confidence=item.confidence,
                source=item.source,
            ))
        return result

    def _extract_water(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        for water in item.water_bodies:
            result.append(TerrainFeature(
                feature_type="water",
                description=f"Water body: {water}",
                confidence=item.confidence,
                source=item.source,
            ))
        return result

    def _extract_roads(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        for road in item.road_network:
            result.append(TerrainFeature(
                feature_type="road",
                description=f"Road: {road}",
                confidence=item.confidence,
                source=item.source,
            ))
        return result

    def _extract_obstacles(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        for obs in item.obstacles:
            result.append(TerrainFeature(
                feature_type="obstacle",
                description=f"Obstacle: {obs}",
                confidence=item.confidence,
                source=item.source,
            ))
        return result

    def _extract_elevation(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        if item.terrain_type:
            result.append(TerrainFeature(
                feature_type="elevation",
                description=f"Terrain type: {item.terrain_type}",
                confidence=item.confidence,
                source=item.source,
            ))
        if item.elevation is not None:
            result.append(TerrainFeature(
                feature_type="elevation",
                description=f"Elevation: {item.elevation}m",
                confidence=item.confidence,
                source=item.source,
            ))
        if item.slope is not None:
            result.append(TerrainFeature(
                feature_type="elevation",
                description=f"Slope: {item.slope} degrees",
                confidence=item.confidence,
                source=item.source,
            ))
        for feat in item.features:
            feat_lower = feat.lower()
            if any(h in feat_lower for h in ("hill", "valley", "ridge", "slope")):
                result.append(TerrainFeature(
                    feature_type="elevation",
                    description=f"Elevation feature: {feat}",
                    confidence=item.confidence,
                    source=item.source,
                ))
        return result

    def _extract_urban(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        for feat in item.features:
            if "urban" in feat.lower() or "building" in feat.lower():
                result.append(TerrainFeature(
                    feature_type="urban",
                    description=f"Urban feature: {feat}",
                    confidence=item.confidence,
                    source=item.source,
                ))
        return result

    def _extract_bridges(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        for feat in item.features:
            if "bridge" in feat.lower() or "culvert" in feat.lower():
                result.append(TerrainFeature(
                    feature_type="bridge",
                    description=f"Crossing: {feat}",
                    confidence=item.confidence,
                    source=item.source,
                ))
        for road in item.road_network:
            if "bridge" in road.lower():
                result.append(TerrainFeature(
                    feature_type="bridge",
                    description=f"Bridge on road: {road}",
                    confidence=item.confidence,
                    source=item.source,
                ))
        return result

    def _extract_soil(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        if item.soil_type:
            result.append(TerrainFeature(
                feature_type="soil",
                description=f"Soil: {item.soil_type}",
                confidence=item.confidence,
                source=item.source,
            ))
        return result

    def _extract_weather(self, item: TerrainData) -> list[TerrainFeature]:
        result: list[TerrainFeature] = []
        if item.weather:
            for key, val in item.weather.items():
                result.append(TerrainFeature(
                    feature_type="weather",
                    description=f"Weather {key}: {val}",
                    confidence=item.confidence,
                    source=item.source,
                ))
        return result
