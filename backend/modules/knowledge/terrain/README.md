# Terrain Knowledge Module

## Purpose

Analyses terrain characteristics from detection data — classifying terrain type, identifying features, assessing mobility, and evaluating visibility. Answers the question: *"What are the terrain conditions in this area?"*

Contains zero GIS, zero mapping engine, zero RAG, zero AI. All terrain sources are injected via `RetrieverInterface`.

## Architecture

```
                       ┌──────────────────────────────────┐
                       │   TerrainKnowledgeService        │
                       │   (public entry point)           │
                       │   implements TerrainModule       │
                       └────────────┬─────────────────────┘
                                    │
                       ┌────────────▼─────────────────────┐
                       │          TerrainEngine           │
                       │   (reasoning orchestration)      │
                       └──┬───────┬──────────┬────────────┘
                          │       │          │
               ┌──────────▼──┐ ┌──▼────────┐ └──▼──────────────┐
               │  Retriever  │ │  Mobility  │  Visibility     │
               │ (abstract)  │ │  Analyzer  │  Analyzer       │
               └─────────────┘ └───────────┘ └─────────────────┘
                          │
               ┌──────────▼──────────────┐
               │ TerrainFeatureBuilder   │
               └─────────────────────────┘
                          │
               ┌──────────▼──────────────┐
               │   ConfidenceScorer      │
               └─────────────────────────┘
                          │
               ┌──────────▼──────────────┐
               │      TerrainAnalysis    │
               │     (contract model)    │
               └─────────────────────────┘
```

## Execution Pipeline

```
DetectionResult
    │
    ▼
TerrainKnowledgeService.analyze_terrain()
    │
    ├─► TerrainEngine.analyze()
    │      │
    │      ├─► Build query from detection objects (types, labels, descriptions)
    │      ├─► Retriever.retrieve(query, context)
    │      │
    │      ├─► TerrainFeatureBuilder.build_features(data)
    │      │      └─► Extracts: vegetation, water, roads, obstacles,
    │      │                    elevation, urban, bridges, soil, weather
    │      │
    │      ├─► MobilityAnalyzer.analyze(features, data)
    │      │      └─► Road access, mobility rating, difficulty, obstacles
    │      │
    │      ├─► VisibilityAnalyzer.analyze(features, data)
    │      │      └─► Visibility, observation, cover, concealment
    │      │
    │      ├─► Classify terrain type (dominant feature)
    │      ├─► Extract elevation
    │      ├─► Collect unique feature names
    │      ├─► ConfidenceScorer.score(detection, terrain, evidence)
    │      │
    │      └─► Assemble TerrainAnalysis
    │
    └─► Return TerrainAnalysis
```

### Step-by-step

1. **Service** receives a `DetectionResult` from the Computer Vision pipeline.
2. **TerrainEngine** builds a query from all detected object types, labels, and descriptions.
3. The **Retriever** is called with the query and context. The default `NullRetriever` returns nothing — replace it with a real terrain source.
4. **TerrainFeatureBuilder** extracts structured `TerrainFeature` objects from the raw terrain data (vegetation, water, roads, obstacles, elevation, urban, bridges, soil, weather).
5. **MobilityAnalyzer** evaluates road access, terrain difficulty, and obstacles to produce a `MobilityAssessment`.
6. **VisibilityAnalyzer** evaluates visibility, observation quality, cover, and concealment to produce a `VisibilityAssessment`.
7. The engine classifies the dominant terrain type, extracts elevation, and collects feature names.
8. **ConfidenceScorer** combines detection confidence, terrain source confidence, and feature quality into a final score.
9. A `TerrainAnalysis` contract is assembled and returned.

## Terrain Pipeline

### Input

`DetectionResult` (from `backend.contracts.models.detection`):
```
image_id: str
timestamp: datetime
objects: list[DetectedObject]    # each with type, label, description, confidence
model_version: str
processing_time_ms: float
```

### Output

`TerrainAnalysis` (from `backend.contracts.models.analysis`):
```
terrain_type: TerrainType        # dominant terrain classification
nearby_features: list[str]       # unique feature type names
visibility: str                  # 'good', 'obscured', or 'blocked'
road_access: bool                # whether road access is available
elevation: float | None          # elevation in metres
reason: str                      # human-readable explanation
```

## Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Service** | Public entry point, DI wiring | Retrieval, features, analysis, scoring |
| **TerrainEngine** | Pipeline orchestration, terrain classification, assembly | Retrieval, scoring |
| **Retriever** | Query terrain sources | Feature extraction, analysis |
| **TerrainFeatureBuilder** | Extract structured features | Confidence, mobility, visibility |
| **MobilityAnalyzer** | Assess road access, difficulty, obstacles | Confidence, visibility |
| **VisibilityAnalyzer** | Assess visibility, cover, concealment | Confidence, mobility |
| **ConfidenceScorer** | Calculate final confidence | Retrieval, features, analysis |
| **NullRetriever** | Default no-op implementation | Real retrieval |

## Internal Data Types

| Type | Fields | Purpose |
|------|--------|---------|
| `TerrainData` | source, terrain_type, elevation, features, road_network, water_bodies, vegetation, obstacles, slope, soil_type, weather, confidence, metadata | Raw retrieved terrain info |
| `TerrainFeature` | feature_type, description, confidence, source | Extracted explainable feature |
| `MobilityAssessment` | road_access, mobility_rating, terrain_difficulty, obstacles, description | Mobility analysis result |
| `VisibilityAssessment` | visibility, observation, cover, concealment, description | Visibility analysis result |

## Future GIS Integration

To add GIS-based retrieval:

1. Implement `RetrieverInterface`:
   ```python
   class GISRetriever(RetrieverInterface):
       def __init__(self, endpoint: str, layer: str):
           # Connect to WMS/WFS/WCS service
           ...
       async def retrieve(self, query: str, context=None) -> RetrievalResult:
           # 1. Parse query as bounding box or coordinate
           # 2. Query GIS layer (roads, waterways, land cover, etc.)
           # 3. Map results to TerrainData items
           ...
   ```

2. The `TerrainConfig` already includes GIS settings:
   - `TERRAIN_GIS_ENDPOINT`
   - `TERRAIN_GIS_LAYER_NAME`
   - `TERRAIN_GIS_TIMEOUT_SECONDS`

3. Inject:
   ```python
   service = TerrainKnowledgeService(retriever=GISRetriever(
       endpoint=terrain_config.gis_endpoint,
       layer=terrain_config.gis_layer_name,
   ))
   ```

## Future Satellite Maps Integration

To add satellite/remote-sensing retrieval:

```python
class SatelliteRetriever(RetrieverInterface):
    def __init__(self, tile_endpoint: str, tile_format: str):
        # Connect to satellite tile server (Sentinel, Landsat, etc.)
        ...
    async def retrieve(self, query: str, context=None) -> RetrievalResult:
        # 1. Resolve query to tile coordinates
        # 2. Download or query satellite tile metadata
        # 3. Extract terrain information (NDVI, elevation, land cover)
        # 4. Map to TerrainData items
        ...
```

Settings in config: `TERRAIN_SATELLITE_TILE_ENDPOINT`, `TERRAIN_SATELLITE_TILE_FORMAT`.

## Future Offline Maps Integration

To add offline map-based retrieval:

```python
class OfflineMapRetriever(RetrieverInterface):
    def __init__(self, map_path: str):
        # Load local GeoJSON, Shapefile, MBTiles, or raster
        ...
    async def retrieve(self, query: str, context=None) -> RetrievalResult:
        # 1. Parse query as coordinate or area
        # 2. Query local map data
        # 3. Map to TerrainData items
        ...
```

## Extension Guide

### How to add a new Retriever

```python
from backend.modules.knowledge.terrain.interfaces import RetrieverInterface, RetrievalResult

class MyTerrainSource(RetrieverInterface):
    async def retrieve(self, query: str, context=None) -> RetrievalResult:
        items = [...]
        return RetrievalResult(items=items, query_time_ms=42.0)
```

Then inject:
```python
service = TerrainKnowledgeService(retriever=MyTerrainSource())
```

### How to add a new Mobility Analyzer

```python
from backend.modules.knowledge.terrain.interfaces import (
    MobilityAnalyzerInterface, MobilityAssessment, TerrainFeature, TerrainData,
)

class AdvancedMobilityAnalyzer(MobilityAnalyzerInterface):
    def analyze(self, features, data) -> MobilityAssessment:
        # Custom mobility logic
        ...
```

Then inject:
```python
engine = TerrainEngine(mobility_analyzer=AdvancedMobilityAnalyzer())
```

### How to add a new Visibility Analyzer

```python
from backend.modules.knowledge.terrain.interfaces import (
    VisibilityAnalyzerInterface, VisibilityAssessment, TerrainFeature, TerrainData,
)

class LidarVisibilityAnalyzer(VisibilityAnalyzerInterface):
    def analyze(self, features, data) -> VisibilityAssessment:
        # LiDAR-based line-of-sight analysis
        ...
```

Then inject:
```python
engine = TerrainEngine(visibility_analyzer=LidarVisibilityAnalyzer())
```

## Design Principles

- **Dependency Injection** — all components receive their dependencies via constructors
- **Single Responsibility** — each class does exactly one thing
- **Open/Closed** — add retrievers and analyzers without modifying existing code
- **Strong Typing** — all inter-component data uses dataclasses or contract models
- **Async-First** — all retrieval and analysis paths are async
- **No Hardcoded Data** — the `NullRetriever` is the only default; real data comes from injected implementations
- **No GIS** — this module provides the framework; GIS/DEM/satellite sources are injected behind interfaces
- **Interchangeable** — the Terrain module mirrors the Friendly and Enemy module patterns, making all three swappable through the orchestration pipeline via their respective contract interfaces
