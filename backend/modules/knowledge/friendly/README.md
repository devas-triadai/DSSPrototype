# Friendly Knowledge Module

## Purpose

Analyses detected objects for matches against known friendly forces. Provides explainable, confidence-scored assessments that answer the question: *"Is this detected unit one of ours?"*

Contains zero AI, zero RAG, zero vector-database logic. All knowledge sources are injected via `RetrieverInterface`.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │   FriendlyKnowledgeService       │
                    │   (public entry point)           │
                    │   implements FriendlyModule      │
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
                    │        FriendlyAnalysis          │
                    │        (contract model)          │
                    └──────────────────────────────────┘
```

## Execution Flow

```
DetectionResult
    │
    ▼
FriendlyKnowledgeService.analyze_friendly()
    │
    ├─► KnowledgeEngine.analyze()
    │      │
    │      ├─► For each DetectedObject:
    │      │      │
    │      │      ├─► Build query from object type + label + description
    │      │      ├─► Retriever.retrieve(query, context)
    │      │      ├─► Filter relevant KnowledgeItems
    │      │      ├─► EvidenceBuilder.build_evidence(obj, items)
    │      │      └─► Collect evidence + track confidence
    │      │
    │      ├─► ConfidenceScorer.score(evidence, detection_conf, knowledge_conf)
    │      │
    │      └─► Assemble FriendlyAnalysis
    │
    └─► Return FriendlyAnalysis
```

### Step-by-step

1. **Service** receives a `DetectionResult` from the Computer Vision pipeline.
2. **KnowledgeEngine** iterates over each `DetectedObject`.
3. For each object, a query string is built from its type, label, and description.
4. The **Retriever** is called with the query. The default `NullRetriever` returns nothing — replace it with a real implementation.
5. Retrieved **KnowledgeItems** are filtered for relevance to the specific object.
6. **EvidenceBuilder** compares the object against each knowledge item, producing explainable `Evidence` objects (vehicle match, marking match, etc.).
7. **ConfidenceScorer** combines detection confidence, knowledge confidence, and evidence strength into a final score.
8. A `FriendlyAnalysis` contract is assembled and returned.

## Knowledge Pipeline

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

`FriendlyAnalysis` (from `backend.contracts.models.analysis`):
```
friendly_match: bool         # whether any friendly match was found
confidence: float            # confidence score [0, 1]
reason: str                  # human-readable explanation
```

## Component Responsibilities

| Component | Responsibility | Does NOT do |
|-----------|---------------|-------------|
| **Service** | Public entry point, DI wiring | Retrieval, scoring, evidence |
| **KnowledgeEngine** | Pipeline orchestration, filtering | Retrieval, confidence calculation |
| **Retriever** | Query knowledge sources | Filtering, scoring |
| **EvidenceBuilder** | Create explainable evidence | Confidence scoring |
| **ConfidenceScorer** | Calculate final confidence | Retrieval, evidence |
| **NullRetriever** | Default no-op implementation | Real retrieval |

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
   service = FriendlyKnowledgeService(retriever=RAGRetriever(llm, embed, store))
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

2. The `FriendlyConfig` already includes vector-database settings:
   - `FRIENDLY_VECTOR_DB_TYPE`
   - `FRIENDLY_VECTOR_DB_ENDPOINT`
   - `FRIENDLY_VECTOR_DB_COLLECTION`
   - `FRIENDLY_VECTOR_DB_TIMEOUT_SECONDS`

3. Inject and use:
   ```python
   service = FriendlyKnowledgeService(retriever=VectorDBRetriever(
       collection=friendly_config.vector_db_collection,
       endpoint=friendly_config.vector_db_endpoint,
   ))
   ```

## Future Knowledge Sources

| Source | Retriever Implementation |
|--------|--------------------------|
| Order of Battle (JSON) | `JSONFileRetriever` — load static OOB data |
| Unit Database | `DatabaseRetriever` — query SQL/NoSQL |
| Military Manuals (PDF) | `PDFRetriever` — extract relevant passages |
| Knowledge Graph | `GraphRetriever` — traverse entity relationships |
| Remote API | `APIRetriever` — query external intelligence service |
| Vector Embeddings | `VectorDBRetriever` — semantic similarity search |

## Extension Guide

### How to add a new Retriever

```python
from backend.modules.knowledge.friendly.interfaces import RetrieverInterface, RetrievalResult

class MyCustomRetriever(RetrieverInterface):
    async def retrieve(self, query: str, context=None) -> RetrievalResult:
        # Your retrieval logic here
        items = [...]
        return RetrievalResult(items=items, query_time_ms=42.0)
```

Then inject:
```python
service = FriendlyKnowledgeService(retriever=MyCustomRetriever())
```

### How to add a new Scoring Algorithm

```python
from backend.modules.knowledge.friendly.interfaces import ConfidenceScorerInterface, Evidence

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
- **No Hardcoded Knowledge** — the `NullRetriever` is the only default; real data comes from injected implementations
- **No AI** — this module provides the framework; AI/LLM/RAG are injected behind interfaces when needed
