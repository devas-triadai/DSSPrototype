# Dataset Pipeline

Orchestration layer for dataset ingestion in DSSPrototype.

## Responsibility

This package is the **single entry point** for dataset ingestion. It
orchestrates five existing modules without duplicating any business
logic:

```
Raw Dataset
    ↓
Dataset Catalog       (backend.dataset_catalog)
    ↓
Ontology Mapping      (backend.ontology_mapping)
    ↓
Dataset Conversion    (backend.dataset_conversion)
    ↓
Dataset Quality       (backend.dataset_quality)
    ↓
Training              (backend.training)
```

No new algorithms. No duplicated validation. No duplicated conversion.
Only orchestration.

## Package Structure

| File             | Responsibility                          |
|------------------|----------------------------------------|
| `__init__.py`    | Package exports                        |
| `config.py`      | Pipeline configuration (timeouts, etc.)|
| `exceptions.py`  | PipelineError hierarchy                |
| `interfaces.py`  | Abstract contracts                     |
| `models.py`      | PipelineResult, DatasetContext, etc.   |
| `workflow.py`    | DatasetWorkflow — stage orchestration  |
| `ingest.py`      | DatasetIngestor — validation + context |
| `service.py`     | PipelineService — public facade        |
| `cli.py`         | Typer CLI                              |
| `README.md`      | This file                              |
| `tests/`         | Comprehensive test suite (100+ tests)  |

## CLI Usage

```bash
# Full pipeline
python -m backend.dataset_pipeline.cli dataset ingest \
    --dataset coco2017 --source /workspace/data/raw/coco2017

# Skip training
python -m backend.dataset_pipeline.cli dataset ingest \
    --dataset coco2017 --source /workspace/data/raw/coco2017 \
    --skip-training

# Skip quality and training
python -m backend.dataset_pipeline.cli dataset ingest \
    --dataset coco2017 --source /workspace/data/raw/coco2017 \
    --skip-quality --skip-training

# Dry run
python -m backend.dataset_pipeline.cli dataset ingest \
    --dataset coco2017 --source /workspace/data/raw/coco2017 \
    --dry-run

# Catalog only
python -m backend.dataset_pipeline.cli dataset catalog \
    --dataset coco2017 --source /workspace/data/raw/coco2017

# Training only
python -m backend.dataset_pipeline.cli dataset train \
    --dataset coco2017
```

## API Usage

```python
from pathlib import Path
from backend.dataset_pipeline.service import PipelineService

service = PipelineService()
result = await service.ingest_dataset(
    dataset_name="coco2017",
    source_path=Path("/workspace/data/raw/coco2017"),
    skip_training=True,
)
print(result.status)
print(result.summary)
```

## Quality Requirements

- Dependency Injection
- Strong typing
- Async-first
- SOLID / Open-Closed
- No global mutable state
- No duplicated business logic
- Reuse existing services exclusively
