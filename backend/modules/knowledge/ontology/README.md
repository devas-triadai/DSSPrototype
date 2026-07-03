# Military Object Ontology Module

## Purpose

The Ontology Layer transforms raw Computer Vision detector labels into
standardized semantic concepts that the Knowledge Base can search more
effectively.

## Architecture

```
DetectedObject(label="truck")
        ↓
    OntologyService.process("truck", confidence=0.85)
        ↓
    ├─ SynonymMapper     → canonical: "truck"
    ├─ CategoryMapper    → parents: ["wheeled_vehicle", "ground_vehicle", "vehicle"]
    ├─ AliasMapper       → equivalents: ["tactical_truck", "logistics_vehicle"]
    └─ ConfidenceAdjuster→ adjusted confidence: 0.80
        ↓
    OntologyResult
        ↓
    KnowledgeRetriever.retrieve(expanded_queries)
        ↓
    KnowledgeResult
```

## Components

### SynonymMapper
Maps equivalent labels (e.g., "automobile", "motorcar") to a canonical
form (e.g., "car").  Never introduces false specificity.

### CategoryMapper
Maps canonical concepts into hierarchical parent categories:
- `truck` → `wheeled_vehicle` → `ground_vehicle` → `vehicle`
- Confidence decays by 10% per level.

### AliasMapper
Resolves alternate names and provides military/civilian equivalents:
- `truck` → military: `tactical_truck`, `logistics_vehicle`
- `truck` → civilian: `semi_truck`, `delivery_truck`

### ConfidenceAdjuster
Ensures adjusted confidence never exceeds detector confidence.
Applies depth penalties for category traversals.

### OntologyEngine
Coordinates all mappers into a single `OntologyResult`.

### OntologyService
Service-level wrapper for dependency injection into the knowledge modules.

## Data Files

Ontology data is stored in `knowledge_base/ontology/`:

- `vehicles.json` — ground vehicles, tanks, APCs, trucks
- `aircraft.json` — fixed-wing, rotary-wing, military aircraft
- `drones.json` — UAVs, UCAVs, quadcopters, loitering munitions
- `weapons.json` — missiles, artillery, guided weapons
- `terrain.json` — terrain types, features, environments

Each entry contains:
- `canonical_name` — standard label
- `aliases` — alternate spellings and synonyms
- `parent_category` — direct parent in taxonomy
- `child_categories` — known sub-types
- `civilian_equivalents` — public-sector analogues
- `military_equivalents` — defense-sector analogues
- `description` — human-readable definition
- `confidence_base` — intrinsic confidence of this concept

## Uncertainty Model

The ontology **never** falsely upgrades a generic label to a specific
military platform:

```
❌ truck → T-90        (never)
❌ car   → BMP-2       (never)
❌ boat  → tank         (never)

✅ truck → cargo_vehicle (0.92)
✅ truck → wheeled_vehicle (0.90)
✅ truck → ground_vehicle (0.85)
✅ truck → vehicle (0.75)
```

## Query Expansion

When a retriever receives query `"truck"`, the ontology layer expands it to:

```
truck
wheeled_vehicle
ground_vehicle
vehicle
tactical_truck
logistics_vehicle
semi_truck
```

All terms are searched; results are ranked by relevance.

## Versioning

Each ontology file carries a `version` and `schema_version`.
The `OntologyEngine.compute_version()` returns a composite string of all
loaded domains.

## Dependency Injection

```python
from backend.modules.knowledge.ontology import OntologyService

# Default construction (loads all standard ontology files)
service = OntologyService()

# Custom engine injection
engine = OntologyEngine(ontology_files={"vehicles": Path("custom.json")})
service = OntologyService(engine=engine)
```

## Integration

The ontology service is injected into the knowledge retrievers:

```python
from backend.modules.knowledge.friendly.service import FriendlyKnowledgeService
from backend.modules.knowledge.ontology import OntologyService

ontology = OntologyService()
service = FriendlyKnowledgeService(ontology_service=ontology)
```

The retriever uses `ontology_service.expand_query(query)` to broaden
knowledge base searches while preserving the original query term.
