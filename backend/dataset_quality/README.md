# Dataset Quality Assurance Layer

Validates canonical DSS datasets for production readiness before model training.

## Architecture

```
Canonical Dataset
       |
       v
+---------------------------+
|    Quality Pipeline       |
|                           |
|  1. Image Validation      |
|  2. Annotation Validation |
|  3. Geometry Validation   |
|  4. Class Validation      |
|  5. Duplicate Detection   |
|  6. Outlier Detection     |
|  7. Imbalance Analysis    |
|  8. Coverage Analysis     |
|  9. Consistency Check     |
| 10. Integrity Check       |
|                           |
|  11. Scoring              |
|  12. Report Generation    |
+---------------------------+
       |
       v
  Quality Report
  (score, issues, verdict)
```

## Components

| Component | Responsibility |
|-----------|---------------|
| `ImageValidator` | Validates image dimensions, format, color space |
| `AnnotationValidator` | Detects negative coords, zero area, out-of-bounds, broken geometries |
| `GeometryValidator` | Validates bbox/polygon/segmentation/OBB geometry |
| `ClassValidator` | Detects unknown classes, ontology mismatch, rare classes, imbalance |
| `DuplicateDetector` | Detects duplicate images (hash-based), duplicate annotations, near-duplicates |
| `OutlierDetector` | Detects extreme aspect ratios, tiny/huge objects, abnormal resolutions |
| `ImbalanceAnalyzer` | Computes class distribution, long-tail ratio, minority classes, augmentation targets |
| `CoverageAnalyzer` | Measures ontology coverage, image coverage, scene/object diversity |
| `ConsistencyChecker` | Verifies metadata, split, ontology, annotation, version consistency |
| `IntegrityChecker` | Verifies checksums, file presence, broken references, version validity |
| `DatasetScorer` | Computes weighted quality score (0-100), letter grade, production-ready verdict |
| `ReportGenerator` | Generates JSON and Markdown quality reports |
| `QualityPipeline` | Orchestrates all checks in sequence |
| `DatasetQualityService` | Public async facade |

## Scoring Methodology

Weights:
- Image Quality: 10%
- Annotation Quality: 25%
- Geometry Quality: 15%
- Ontology Coverage: 15%
- Balance: 10%
- Integrity: 15%
- Consistency: 10%

Letter Grade:
- A >= 90: Excellent
- B >= 75: Good
- C >= 60: Fair
- D >= 40: Poor
- F < 40: Failing

Production Ready: score >= 75 AND no ERROR-level issues.

## Usage

```python
from backend.dataset_quality.service import DatasetQualityService

service = DatasetQualityService()
report = await service.run_pipeline(dataset)

print(report.overall_score.overall)
print(report.overall_score.letter_grade)
print(report.overall_score.production_ready)

for issue in report.all_issues:
    print(f"[{issue.severity.value}] {issue.message}")
```

## Configuration

Set via env vars with prefix `DATASET_QUALITY_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_QUALITY_MIN_IMAGE_WIDTH` | 32 | Minimum acceptable image width |
| `DATASET_QUALITY_MIN_IMAGE_HEIGHT` | 32 | Minimum acceptable image height |
| `DATASET_QUALITY_MAX_IMAGE_WIDTH` | 10000 | Maximum acceptable image width |
| `DATASET_QUALITY_MAX_IMAGE_HEIGHT` | 10000 | Maximum acceptable image height |
| `DATASET_QUALITY_MIN_OBJECT_AREA` | 4.0 | Minimum object area in pixels |
| `DATASET_QUALITY_MAX_ASPECT_RATIO` | 20.0 | Maximum acceptable aspect ratio |
| `DATASET_QUALITY_RARE_CLASS_THRESHOLD` | 0.01 | Proportion below which a class is rare |
| `DATASET_QUALITY_DUPLICATE_IOU_THRESHOLD` | 0.95 | IoU threshold for exact duplicates |
| `DATASET_QUALITY_NEAR_DUPLICATE_IOU_THRESHOLD` | 0.85 | IoU threshold for near-duplicates |
| `DATASET_QUALITY_OUTLIER_STD_DEV_THRESHOLD` | 3.0 | Std dev threshold for outlier detection |
| `DATASET_QUALITY_MIN_CLASS_SAMPLES` | 5 | Minimum samples per class |
| `DATASET_QUALITY_OUTPUT_DIR` | quality_reports | Output directory for reports |
| `DATASET_QUALITY_STRICT_MODE` | false | Fail on warnings in strict mode |

## Integration

### With Dataset Conversion Pipeline

```python
from backend.dataset_conversion.service import DatasetConversionService
from backend.dataset_quality.service import DatasetQualityService

converter = DatasetConversionService()
quality = DatasetQualityService()

result = await converter.convert_dataset(...)
report = await quality.run_pipeline(result.dataset)
```

### With Training Platform

The quality report's `production_ready` flag gates training:

```python
if report.overall_score.production_ready:
    train_model(dataset)
else:
    log_quality_block(report)
```
