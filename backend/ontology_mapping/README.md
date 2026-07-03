# Ontology Mapping Layer

Translates labels from **any external dataset** into the **canonical DSS ontology**.

External datasets **MUST NEVER** directly enter the training pipeline. Every dataset is first translated through this layer using declarative mapping rules.

## Architecture

```
Dataset Label  ──►  Mapping Engine  ──►  Ontology Value
                            │
                    ┌───────┴───────┐
                    │               │
              Ontology          Mapping
              Resolver          Registry
                    │               │
                    └───────┬───────┘
                            │
                    Conflict Resolver
                            │
                    ┌───────┴───────┐
                    │               │
               Validator       Statistics
                    │               │
                    └───────┬───────┘
                            │
                          Export
```

## Quick Start

```python
from backend.ontology_mapping import (
    OntologyMappingService,
    DatasetProfile,
    MappingRule,
    MatchType,
)

service = OntologyMappingService()

# 1. Register a dataset with mapping rules
profile = DatasetProfile(
    dataset_name="coco",
    version="2017",
    label_count=80,
    labels=["truck", "car", "person", ...],
)

rules = [
    MappingRule(
        dataset_name="coco",
        source_label="truck",
        canonical_value="ground_vehicle.truck",
        match_type=MatchType.EXACT,
    ),
    MappingRule(
        dataset_name="coco",
        source_label="car",
        canonical_value="ground_vehicle.car",
        match_type=MatchType.EXACT,
    ),
]

await service.register_dataset(profile, rules)

# 2. Translate labels
result = await service.map_label("coco", "truck")
print(result.canonical_value)   # "ground_vehicle.truck"
print(result.canonical_name)    # "Truck"
print(result.confidence)        # 1.0
```

## Package Structure

| File | Responsibility |
|---|---|
| `config.py` | Environment-driven configuration (`ONTOLOGY_MAPPING_*` prefix) |
| `exceptions.py` | Exception hierarchy rooted at `MappingError` |
| `interfaces.py` | Abstract interfaces (ABCs) for all public capabilities |
| `models.py` | All frozen Pydantic data models |
| `registry.py` | In-memory registry of datasets and mapping rules |
| `mapping_engine.py` | Label-to-ontology translation with multi-strategy matching |
| `ontology_resolver.py` | Ontology tree builder and traversal (built from ObjectType) |
| `conflict_resolver.py` | Conflict detection and deterministic resolution |
| `mapping_validator.py` | Validation of ontology structure and mapping rules |
| `mapping_statistics.py` | Coverage, health, and compatibility statistics |
| `mapping_version.py` | Semantic version management for mapping states |
| `dataset_mapper.py` | Dataset-level mapping lifecycle orchestration |
| `exporter.py` | JSON, YAML, CSV export |
| `service.py` | Public API facade |

## Matching Strategies

The `MappingEngine` attempts matches in order of precision:

1. **Exact** — character-for-character match (confidence 1.0)
2. **Case-insensitive** — lowercase match (confidence 0.95)
3. **Alias** — registered alias table lookup (confidence 0.9)
4. **Plural** — singularization normalization (confidence 0.85)
5. **Synonym** — registered synonym table lookup (confidence 0.7)
6. **Regex** — pattern-based matching (configurable confidence)
7. **Fuzzy** — character-set similarity against ontology names (variable confidence)

## Configuration

All settings use `ONTOLOGY_MAPPING_` prefix:

| Variable | Default | Description |
|---|---|---|
| `ONTOLOGY_MAPPING_VERSION` | `1.0.0` | Mapping layer version |
| `ONTOLOGY_MAPPING_STRICT_MODE` | `True` | Reject unknown labels in strict mode |
| `ONTOLOGY_MAPPING_VALIDATION_ENABLED` | `True` | Validate on all operations |
| `ONTOLOGY_MAPPING_CACHE_ENABLED` | `True` | Enable result caching |
| `ONTOLOGY_MAPPING_CACHE_TTL_SECONDS` | `3600` | Cache TTL |
| `ONTOLOGY_MAPPING_LOG_LEVEL` | `INFO` | Logging verbosity |

## Integration Points

- **Dataset Catalog**: Register dataset profiles with `register_dataset()`
- **Dataset Intelligence**: Use `map_dataset()` to translate annotations
- **Training Platform**: Use exported mappings in `map_labels()` for dataloader integration
- **Any future dataset**: Add mapping rules via `register_dataset()`, no downstream changes needed

## Key Design Decisions

1. **Frozen ontology** — The ontology tree is built from the frozen `ObjectType` enum at startup
2. **No global state** — All state is in `MappingRegistry`, injected into services
3. **Dependency injection** — Every component accepts its dependencies as constructor parameters
4. **Async-first** — Public API is fully async; internal components are sync
5. **Immutable models** — All Pydantic models use `frozen=True`
6. **Open/Closed** — New datasets are added by registering new rules; no code changes
