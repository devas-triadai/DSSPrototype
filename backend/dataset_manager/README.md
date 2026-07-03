# Dataset Management Platform

Single source of truth for all Computer Vision training datasets.

## Architecture

```
dataset_manager/
├── __init__.py      # Package exports
├── config.py        # DatasetManagerConfig (pydantic-settings)
├── interfaces.py    # All ABC contracts
├── models.py        # Pydantic frozen models
├── registry.py      # DatasetRegistry (in-memory, swappable)
├── validator.py     # DatasetValidator (12 validation checks)
├── statistics.py    # StatisticsEngine (comprehensive metrics)
├── splitter.py      # DatasetSplitter (deterministic TVT split)
├── versioning.py    # DatasetVersioning (semantic versions)
├── exporter.py      # YoloExporter, CocoExporter, PascalVocExporter
├── quality.py       # QualityEngine (multi-factor quality score)
├── metadata.py      # MetadataGenerator (auto-generates metadata.json)
├── loader.py        # DatasetLoader (image/annotation discovery)
├── checksum.py      # ChecksumGenerator (SHA256)
├── service.py       # DatasetManagementService (DI coordinator)
└── exceptions.py    # DatasetError hierarchy
```

## Key Design Decisions

- **Dependency Injection**: All components are injected via constructor, following DSS pattern
- **Open/Closed**: Add new validators, exporters, or quality metrics by implementing interfaces
- **Frozen models**: All data models use `ConfigDict(frozen=True)` for immutability
- **Async-first**: Core operations are async where beneficial
- **No global state**: Registry is per-instance, fully swappable

## Dataset Registry

| Field | Type | Description |
|---|---|---|
| dataset_id | str | Unique identifier |
| dataset_name | str | Human-readable name |
| dataset_version | str | Semantic version |
| dataset_type | str | raw / annotated / split |
| validation_status | str | pending / passed / failed |
| quality_score | float | 0.0 - 1.0 |
| checksum | DatasetChecksum | SHA256 |

## Validation Checks

1. Missing images
2. Missing labels
3. Empty annotations
4. Duplicate images
5. Duplicate annotations
6. Corrupted files (zero-byte)
7. Unsupported extensions
8. Invalid bounding boxes
9. Negative coordinates
10. Class mismatches
11. Orphan labels
12. Invalid metadata

## Quality Score (weighted)

| Factor | Weight | Description |
|---|---|---|
| Annotation completeness | 25% | Fraction of images with annotations |
| Dataset balance | 15% | Inverse class imbalance ratio |
| Image quality metadata | 10% | Image quality signals |
| Duplicate percentage | 15% | Inverse of duplicate ratio |
| Missing labels | 10% | Fraction of images missing labels |
| Validation score | 25% | Fraction of validation checks passed |

## Export Formats

- **YOLO**: images/ + labels/ + data.yaml
- **COCO**: annotations.json (images/annotations/categories)
- **Pascal VOC**: One XML per image

Add new formats by implementing `DatasetExporterInterface`.
