# Enemy Knowledge Module

## Purpose

Analyses detected objects for matches against known enemy forces. Provides explainable, confidence-scored assessments that answer the question: *"Is this detected unit one of theirs?"*

Contains zero AI, zero RAG, zero vector-database logic. All intelligence sources are injected via `RetrieverInterface`.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │    EnemyKnowledgeService         │
                    │   (public entry point)           │
                    │   implements EnemyModule         │
                    └────────────┬─────────────────────┘
                                 │
                    ┌────────────▼─────────────────────┐
                    │        KnowledgeEngine           │
                    │   (reasoning orchestration)      │
                    └────┬──────────┬──────────────────┘
                         │          │
              ┌──────────▼──┐  ┌────▼──────────────┐
              │  Retriever  │  │  ConfidenceScorer │
              │ (abstract)  │  │  (replaceable)    │
              └─────────────┘  └───────────────────┘
                         │          │
              ┌──────────▼──┐       │
              │EvidenceBuilder│      │
              └─────────────┘       │
                                    │
                    ┌───────────────▼──────────────────┐
                    │         EnemyAnalysis            │
                    │        (contract model)          │
                    └──────────────────────────────────┘
```

## Execution Flow

```
DetectionResult
    │
    ▼
EnemyKnowledgeService.analyze_enemy()
    │
    ├─► KnowledgeEngine.analyze()
    │      │
    │      ├─► For each DetectedObject:
    │      │      │
    │      │      ├─► Build query from object type + label + description
    │      │      ├─► Retriever.retrieve(query, context)
    │      │      ├─► Filter relevant KnowledgeItems
    │      │      ├─► EvidenceBuilder.build_evidence(obj, items)
    │      │      └─► Collect evidence + equipment IDs + track confidence
    │      │
    │      ├─► ConfidenceScorer.score(evidence, detection_conf, knowledge_conf)
    │      │
    │      └─► Assemble EnemyAnalysis
    │
    └─► Return EnemyAnalysis
```

### Step-by-step

1. **Service** receives a `DetectionResult` from the Computer Vision pipeline.
2. **KnowledgeEngine** iterates over each `DetectedObject`.
3. For each object, a query string is built from its type, label, and description.
4. The **Retriever** is called with the query. The default `NullRetriever` returns nothing — replace it with a real implementation.
5. Retrieved **KnowledgeItems** are filtered for relevance to the specific object (equipment, markings, country, tactical role).
6. **EvidenceBuilder** compares the object against each intelligence item, producing explainable `Evidence` (vehicle ID, platform match, country attribution, weapon system, camouflage, capability, threat indicator, tactical role).
7. **ConfidenceScorer** combines detection confidence, intelligence confidence, and evidence strength into a final score.
8. Equipment identifiers are collected from matched intelligence items.
9. An `EnemyAnalysis` contract is assembled and returned.

## Intelligence Pipeline

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

`EnemyAnalysis` (from `backend.contracts.models.analysis`):
```
enemy_match: bool                # whether any enemy match was found
confidence: float                # confidence score [0, 1]
possible_equipment: str | None   # identified equipment (comma-separated)
reason: str                      # human-readable explanation
```

## Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Service** | Public entry point, DI wiring | Retrieval, scoring, evidence |
| **KnowledgeEngine** | Pipeline orchestration, filtering, equipment collection | Retrieval, confidence calculation |
| **Retriever** | Query intelligence sources | Filtering, scoring |
| **EvidenceBuilder** | Create explainable evidence | Confidence scoring |
| **ConfidenceScorer** | Calculate final confidence | Retrieval, evidence |
| **NullRetriever** | Default no-op implementation | Real retrieval |

## Evidence Types

| Type | Description | Default Weight |
|------|-------------|----------------|
| `vehicle_id` | Equipment matches detected type | 0.85 |
| `platform_match` | Specific platform identification | 0.90 |
| `country_attribution` | Country of origin match | 0.75 |
| `weapon_system` | Known weapon system identified | 0.80 |
| `camouflage_match` | Camouflage or marking match | 0.70 |
| `capability_match` | Known enemy capability | 0.60 |
| `threat_indicator` | Known threat indicator observed | 0.95 |
| `tactical_role` | Tactical role matches context | 0.65 |

## Future RAG Integration

To add RAG-based retrieval:

1. Implement `RetrieverInterface` with a RAG pipeline:
   ```python
   class RAGRetriever(RetrieverInterface):
       def __init__(self, llm: Any, embedding_model: Any, vector_store: Any):
           ...
       async def retrieve(self, query: str, context=None) -> RetrievalResult:
           # 1. Embed query
           # 2. Vector search
           # 3. LLM rerank
           # 4. Map results to KnowledgeItem list
           ...
   ```

2. Inject it into the service:
   ```python
   service = EnemyKnowledgeService(retriever=RAGRetriever(llm, embed, store))
   ```

The `KnowledgeEngine`, `EvidenceBuilder`, and `ConfidenceScorer` remain unchanged. RAG only replaces the retrieval step.

## Future Vector Database Integration

To add a vector database backend:

1. Implement `RetrieverInterface`:
   ```python
   class VectorDBRetriever(RetrieverInterface):
       def __init__(self, collection_name: str, endpoint: str):
           # Connect to Pinecone, Weaviate, Chroma, Qdrant, etc.
           ...
       async def retrieve(self, query: str, context=None) -> RetrievalResult:
           # 1. Generate embedding for query
           # 2. Query vector DB top-k
           # 3. Map results to KnowledgeItem list
           ...
   ```

2. The `EnemyConfig` already includes vector-database settings:
   - `ENEMY_VECTOR_DB_TYPE`
   - `ENEMY_VECTOR_DB_ENDPOINT`
   - `ENEMY_VECTOR_DB_COLLECTION`
   - `ENEMY_VECTOR_DB_TIMEOUT_SECONDS`

3. Inject and use:
   ```python
   service = EnemyKnowledgeService(retriever=VectorDBRetriever(
       collection=enemy_config.vector_db_collection,
       endpoint=enemy_config.vector_db_endpoint,
   ))
   ```

## Future Intelligence Sources

| Source | Retriever Implementation |
|--------|--------------------------|
| Threat Library | `ThreatLibraryRetriever` — query known threat database |
| Equipment Catalogue | `EquipmentCatalogueRetriever` — match vehicle/weapon specs |
| Military Manuals (PDF) | `PDFRetriever` — extract relevant passages |
| Open-source Intelligence | `OSINTRetriever` — scrape or query OSINT feeds |
| Knowledge Graph | `GraphRetriever` — traverse entity relationships |
| Remote API | `APIRetriever` — query external intelligence service |
| Vector Embeddings | `VectorDBRetriever` — semantic similarity search |

## Extension Guide

### How to add a new Retriever

```python
from backend.modules.knowledge.enemy.interfaces import RetrieverInterface, RetrievalResult

class MyIntelligenceSource(RetrieverInterface):
    async def retrieve(self, query: str, context=None) -> RetrievalResult:
        # Your retrieval logic here
        items = [...]
        return RetrievalResult(items=items, query_time_ms=42.0)
```

Then inject:
```python
service = EnemyKnowledgeService(retriever=MyIntelligenceSource())
```

### How to add a new Scoring Algorithm

```python
from backend.modules.knowledge.enemy.interfaces import ConfidenceScorerInterface, Evidence

class BayesianScorer(ConfidenceScorerInterface):
    def score(self, evidence, detection_confidence, knowledge_confidence) -> float:
        # Bayesian fusion logic
        ...
```

Then inject:
```python
engine = KnowledgeEngine(confidence_scorer=BayesianScorer())
```

## Design Principles

- **Dependency Injection** — all components receive their dependencies via constructors
- **Single Responsibility** — each class does exactly one thing
- **Open/Closed** — add retrievers and scorers without modifying existing code
- **Strong Typing** — all inter-component data uses dataclasses or contract models
- **Async-First** — all retrieval and analysis paths are async
- **No Hardcoded Intelligence** — the `NullRetriever` is the only default; real data comes from injected implementations
- **No AI** — this module provides the framework; AI/LLM/RAG are injected behind interfaces when needed
- **Interchangeable** — the Enemy module mirrors the Friendly module structure, making both swappable through the orchestration pipeline via their respective contract interfaces
