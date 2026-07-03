# Dataset Intelligence & Standardization Pipeline

The **only** entry point for all future datasets into DSSPrototype.

## Overview

Every dataset imported into the system must pass through this pipeline before it
becomes available for training. The pipeline enforces a single canonical
representation, maps classes to the DSS ontology, detects duplicates, harmonizes
naming conventions, assesses quality, and produces a training-ready export.

## Pipeline Stages

```
Import
  ↓
Format Detection
  ↓
Validation
  ↓
Metadata Extraction
  ↓
Class Extraction
  ↓
Normalization
  ↓
Ontology Mapping
  ↓
Duplicate Detection
  ↓
Class Harmonization
  ↓
Quality Analysis
  ↓
Statistics Generation
  ↓
Dataset Version Creation
  ↓
Ready For Training
```

## Module Structure

| File | Responsibility |
|------|--------------|
| `__init__.py` | Public API exports |
| `config.py` | Configuration (paths, thresholds, formats) |
| `interfaces.py` | Abstract contracts (ABC) for every component |
| `models.py` | Immutable Pydantic models for all pipeline artifacts |
| `parser.py` | Pluggable format parser registry + concrete parsers |
| `importer.py` | Import orchestrator (detect format → parse → RawDataset) |
| `validator.py` | Structural & semantic validation engine |
| `normalizer.py` | Canonical representation (names, bbox, filenames) |
| `ontology_mapper.py` | Maps classes to DSS ontology via OntologyService |
| `duplicate_detector.py` | Filename, hash, metadata, near-duplicate detection |
| `class_harmonizer.py` | Unifies class names across datasets |
| `merger.py` | Merges multiple datasets with provenance |
| `splitter.py` | Train/validation/test splitting (stratified) |
| `statistics.py` | Comprehensive statistics engine |
| `quality.py` | Composite quality score + gating |
| `exporter.py` | YOLO, COCO, VOC exporters + plugin registry |
| `registry.py` | Persistent JSON-backed dataset registry |
| `service.py` | Public orchestrator (DatasetIntelligenceService) |
| `exceptions.py` | Typed exception hierarchy |
| `README.md` | This file |

## Supported Import Formats

- **YOLO** — images + `labels/*.txt` + optional `classes.txt` / `data.yaml`
- **COCO** — `instances.json` or `annotations.json` + `images/`
- **Pascal VOC** — `JPEGImages/` + `*.xml` annotations
- **CSV** — `*.csv` with columns for filename, class, bbox coordinates
- **JSON** — generic JSON array of image records
- **GeoJSON** — feature collection with bbox properties

New formats are plug-and-play: implement `FormatParserInterface` and register.

## Ontology Mapping

All class names are resolved through the existing DSS Ontology Engine (Prompt 17).
Mappings are **configurable and versioned**, not hardcoded.

Examples:
- `tank` → `main_battle_tank`
- `MBT` → `main_battle_tank`
- `battle tank` → `main_battle_tank`
- `BMP II` / `BMP-2` → `bmp_2`
- `truck` / `lorry` → `cargo_truck`

## Provenance Tracking

Every image retains:
- source dataset
- original class
- normalized class
- ontology class
- import timestamp
- license
- dataset version
- checksum (SHA256)

## Quality Gates

| Metric | Default Threshold |
|--------|-------------------|
| Minimum quality score | 0.6 |
| Maximum duplicate ratio | 0.15 |
| Minimum images | 10 |
| Minimum annotations | 10 |
| Minimum classes | 1 |
| Maximum class imbalance | 10.0 |
| Minimum ontology coverage | 0.5 |

## Export Formats

- **YOLO** — `data.yaml` + `images/` + `labels/`
- **COCO** — `images/` + `instances.json`
- **Pascal VOC** — `JPEGImages/` + `Annotations/` + XML

## Integration

### Dataset Manager
After successful processing, the service calls `DatasetManagementService.register_dataset()`
with a `DatasetInfo` populated from the processed dataset. No modifications to the
Dataset Manager are required.

### Training Platform
The `TrainerInterface.train()` method accepts `dataset_metadata`. The service produces
`ProcessedDataset` objects that can be converted to `DatasetInfo` / `DatasetMetadata` and
passed to the Training Platform. Only processed datasets in `ready_for_training/` are
ever consumed by the trainer.

## Usage

```python
from pathlib import Path
from backend.dataset_intelligence.service import DatasetIntelligenceService

service = DatasetIntelligenceService()
processed = service.import_dataset(
    source_path=Path("datasets/incoming/my_raw_dataset"),
    dataset_name="my_dataset",
)
print(processed.quality.quality_score)
print(processed.export_result.output_dir)
```

## Design Principles

- **Dependency Injection** — every component depends only on interfaces
- **SOLID** — single responsibility, open/closed, Liskov substitution
- **No global mutable state** — all state is injected or persisted
- **Fully testable** — every component can be mocked behind its interface
- **Async-ready** — I/O boundaries accept async where beneficial
- **Extensible** — parsers and exporters are plugin-based
