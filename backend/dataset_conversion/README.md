# Dataset Conversion Pipeline

Converts arbitrary public datasets into a unified canonical DSS training format.

## Architecture

```
External Dataset
       │
       ▼
 Dataset Loader ────► Annotation Loader
       │                      │
       ▼                      ▼
 Image Converter    Ontology Adapter ←── Ontology Mapping Layer
       │                      │
       ▼                      ▼
 ImageInfo         Annotation Converter
       │                      │
       ▼                      ▼
   ┌──────────────────────────────┐
   │       CanonicalDataset        │
   └──────────────────────────────┘
       │              │       │
       ▼              ▼       ▼
 Dataset Merger   Splitter  Validator
       │              │       │
       ▼              ▼       ▼
   ┌──────────────────────────────┐
   │     Export / Manifest / Stats│
   └──────────────────────────────┘
```

## Components

### Dataset Loader
- Loads datasets from disk in multiple formats (COCO JSON, YOLO TXT, Pascal VOC XML, Open Images CSV)
- Lazy-loading support
- Returns strongly typed `LoadResult`

### Annotation Loader
- Parses annotation files into `SourceAnnotation` models
- Plugin-based for new formats
- Never returns raw dictionaries

### Ontology Adapter
- Wraps the Ontology Mapping Service
- Translates source labels to canonical DSS ontology labels
- Strict mode rejects unknown labels

### Geometry Converter
- Converts between geometry formats (bbox, polygon, OBB, normalized)
- Supports pixel and normalized coordinate systems

### Annotation Converter
- Converts `SourceAnnotation` → `CanonicalAnnotation`
- Preserves class, geometry, confidence, metadata
- Denormalizes coordinates as needed

### Image Converter
- Validates images exist and meet constraints
- Standardizes format, color space, metadata

### Dataset Merger
- Merges multiple `CanonicalDataset` into one
- Prevents duplicate image/annotation IDs
- Tracks provenance with source dataset names

### Dataset Splitter
- Splits into train/val/test
- Supports random, stratified, and class-balanced strategies
- Deterministic seed for reproducibility

### Dataset Validator
- 8 validation checks: images, annotations, geometry, ontology, duplicates, orphans, class coverage, split integrity
- Returns `ValidationReport` with errors and warnings

### Statistics Generator
- Computes: image counts, annotation counts, class balance, bbox statistics, coverage report

### Manifest Builder
- Versioned `DatasetManifest` with checksums, ontology version, pipeline version

### Dataset Exporter
- Export to canonical DSS JSON, COCO JSON, YOLO TXT format

### Conversion Pipeline
- Orchestrates the full conversion lifecycle: load → convert → validate → report

### Service
- Public async facade with all methods

## Usage

```python
from backend.dataset_conversion import DatasetConversionService

service = DatasetConversionService()

# Load
load_result = await service.load_dataset("path/to/coco.json", "coco_json")

# Convert
dataset = await service.convert_dataset(load_result, "my_dataset")

# Validate
report = await service.validate_dataset(dataset)

# Statistics
stats = await service.dataset_statistics(dataset)

# Split
from backend.dataset_conversion.models import SplitConfig, SplitStrategy
split = await service.split_dataset(dataset, SplitConfig(
    strategy=SplitStrategy.RANDOM,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    seed=42,
))

# Merge
merged = await service.merge_datasets([split.train, split.val])

# Export
result = await service.export_dataset(dataset, "coco_json", "/output/path")

# Manifest
manifest = await service.build_manifest(dataset, stats)

# Full pipeline
convert_result = await service.run_pipeline(
    "path/to/source", "coco_json", "my_dataset", "/output/path",
    data_dir="/path/to/images",
)
```

## Configuration

Environment variables with `DATASET_CONVERSION_` prefix:

| Variable | Default | Description |
|---|---|---|
| `DATASET_CONVERSION_VERSION` | `1.0.0` | Pipeline version |
| `DATASET_CONVERSION_STRICT_MODE` | `true` | Fail on unmapped labels |
| `DATASET_CONVERSION_DEFAULT_TRAIN_RATIO` | `0.7` | Default training split ratio |
| `DATASET_CONVERSION_DEFAULT_SEED` | `42` | Default random seed |

## Integration

### Ontology Mapping Layer
The `OntologyAdapter` wraps `OntologyMappingService`. Source labels are translated through the same 7-strategy mapping engine (exact → case-insensitive → alias → plural → synonym → regex → fuzzy).

### Dataset Catalog
The `DatasetManifest` can be registered with the Dataset Catalog for version tracking and discovery.

### Dataset Intelligence
Converted datasets feed directly into Dataset Intelligence for analysis.

### Training Platform
Export to canonical format for direct ingestion by the Training Platform dataloader.
