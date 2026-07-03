# Dataset Acquisition & Curation System

Sits **upstream** of the Dataset Intelligence Pipeline. It discovers, profiles,
scores, prioritizes, and curates candidate datasets **before** they enter the
import pipeline.

## Architecture

```
                    ┌──────────────────────────────┐
                    │   Dataset Catalog Service     │
                    │   (orchestrator)               │
                    └──────┬──────┬──────┬──────┬───┘
                           │      │      │      │
              ┌────────────┘      │      │      └────────────┐
              ▼                   ▼      ▼                   ▼
    ┌─────────────────┐  ┌──────────┐  ┌────────┐  ┌──────────────┐
    │  DatasetProfiler │  │ Catalog  │  │  Gap   │  │ CurationSvc  │
    │  SourceRegistry  │  │ Coverage │  │Analyzer│  │              │
    │  ClassTaxonomy   │  │Analyzer  │  │        │  │              │
    │  LicenseManager  │  │          │  │        │  │              │
    └─────────────────┘  └──────────┘  └────────┘  └──────────────┘
```

## Components

| Component | Responsibility |
|-----------|---------------|
| `Catalog` | Persistent JSON-backed store for catalog entries |
| `SourceRegistry` | Tracks dataset sources with reliability scoring |
| `DatasetProfiler` | Profiles candidate datasets by inspecting their contents |
| `ClassTaxonomy` | Manages a hierarchical military-domain taxonomy tree |
| `LicenseManager` | Classifies licenses, computes risk, checks compatibility |
| `CoverageAnalyzer` | Analyzes taxonomy coverage across catalog entries |
| `GapAnalyzer` | Identifies and prioritizes coverage gaps |
| `RecommendationEngine` | Scores and recommends datasets for acquisition |
| `AcquisitionPlanner` | Plans and tracks dataset acquisition projects |
| `CurationService` | Manages the review/approval workflow |
| `DatasetCatalogService` | Public orchestrator — the only entry point |

## Scoring Dimensions

Scores are combined via configurable weights (`DC_WEIGHT_*`):

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Quality | 0.30 | Format consistency, missing/corrupt images |
| Coverage | 0.25 | Taxonomy node coverage ratio |
| Diversity | 0.15 | Class distribution uniformity |
| License | 0.15 | Inverse of license risk score |
| Source Reliability | 0.15 | Source fetch success rate |

## Configuration

All values overridable via environment variables with prefix `DC_`:

```bash
DC_MAX_ACTIVE_ACQUISITIONS=10
DC_WEIGHT_QUALITY=0.25
DC_ALLOWED_LICENSES='["cc0","mit"]'
```

## Usage

```python
from backend.dataset_catalog import DatasetCatalogService

svc = DatasetCatalogService()

# Discover and profile a candidate dataset
entry = svc.discover(
    path=Path("/path/to/dataset"),
    source_id="src_001",
    source_type="local",
    curator="analyst_01",
)

# Get catalog-wide coverage analysis
coverage = svc.get_catalog_coverage()

# Identify gaps
gaps = svc.get_gap_analysis()

# Get recommendations
recommendations = svc.get_recommendations(domain="military", limit=10)

# Create acquisition plan
plan = svc.create_acquisition_plan(
    entries=[entry.entry_id],
    priority=0.85,
    notes="High-priority ground vehicle dataset",
)

# Curate the entry
record = svc.submit_for_curation(entry.entry_id, curator="analyst_01")
record = svc.approve_curation(record.record_id, reviewer="manager_01")
```
