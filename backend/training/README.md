# Enterprise Training Platform

Manages the complete lifecycle of computer vision model training — from dataset intake through model export and registry.

## Architecture Overview

```
Dataset Quality
     │
     ▼
DatasetLoader ─── rejects if not production_ready
     │
     ▼
AugmentationPipeline ─── Flip, Rotate, Scale, Crop, Brightness, Contrast, Blur, Noise, Mosaic, MixUp
     │
     ▼
TrainingValidator ─── validates config before training
     │
     ▼
Trainer ─── orchestrates full training lifecycle via DI
     │
     ├── TrainingBackendInterface ─── YOLO, RT-DETR, DETR, etc.
     ├── ExperimentManager ─── experiment lifecycle
     ├── ModelRegistry ─── persistent model catalog
     ├── CheckpointManager ─── best/last/epoch checkpoints
     ├── MetricsManager ─── per-epoch metric snapshots
     ├── HistoryManager ─── training history for plotting
     ├── EvaluationEngine ─── validation, test, benchmark
     ├── ExportPipeline ─── ONNX, TorchScript, OpenVINO
     └── Scheduler ─── cosine, step, linear, polynomial
     │
     ▼
TrainingPipeline ─── single orchestrator for full lifecycle
     │
     ▼
TrainingService ─── public async facade
```

## Package Structure

| File | Responsibility |
|------|---------------|
| `__init__.py` | Public API — exports all symbols |
| `config.py` | `TrainingConfig` via pydantic-settings (`TR_` prefix) |
| `exceptions.py` | 9 exception classes rooted at `TrainingError` |
| `interfaces.py` | 13 ABC interfaces for all components |
| `models.py` | 16 frozen Pydantic models |
| `dataset_loader.py` | Loads production-approved datasets from Dataset Quality |
| `augmentation.py` | Configurable augmentation pipeline (10 transforms) |
| `hyperparameter_manager.py` | Named training profiles (fast, balanced, accurate, tiny) |
| `validator.py` | Pre-training configuration validation |
| `trainer.py` | Training lifecycle coordinator via DI |
| `training_pipeline.py` | Single orchestrator (10 stages) |
| `service.py` | Public facade with DI |
| `checkpoint.py` | Checkpoint save/load/prune lifecycle |
| `evaluator.py` | Validation, test, benchmark scaffolding |
| `experiment.py` | Experiment CRUD with JSON persistence |
| `exporter.py` | Export to ONNX, TorchScript, OpenVINO |
| `history.py` | Training history for plotting |
| `metrics.py` | Per-epoch metric recording and querying |
| `registry.py` | Persistent model registry |
| `scheduler.py` | Cosine, step, linear, polynomial LR schedules |
| `callbacks.py` | Callback system (on_epoch_start/end, etc.) |
| `early_stopping.py` | Early stopping with patience and restore-best |

### Backends

| File | Responsibility |
|------|---------------|
| `backends/__init__.py` | Backend package exports |
| `backends/registry.py` | `TrainingBackendRegistry` — register/create backends by name |
| `backends/yolo_backend.py` | YOLO training backend (Ultralytics) |

## End-to-End Training Lifecycle

```
TrainingService.run_pipeline(config, augmentation_config, dataset_metadata)
  │
  ├── Stage  1: Validate Dataset ─── DatasetLoader rejects non-production-ready
  ├── Stage  2: Load Dataset ─────── Resolves dataset YAML path
  ├── Stage  3: Augment ──────────── Creates augmentation pipeline from config
  ├── Stage  4: Validate Config ──── TrainingValidator checks all hyperparameters
  │
  └── Trainer.train(config, dataset_metadata)
        │
        ├── Create experiment
        ├── Register model (status=training)
        ├── Initialize training backend
        │
        └── For each epoch:
              ├── Scheduler.get_lr(epoch) → learning rate
              ├── TrainingBackend.train_epoch() → metrics
              ├── CheckpointManager.save_checkpoint() (every save_interval)
              ├── TrainingBackend.validate() → evaluation metrics
              ├── EvaluationEngine.validate() → persisted result
              ├── MetricsManager.record() → per-epoch metrics
              ├── HistoryManager.record_entry() → plot-ready history
              └── EarlyStopping.check() → break if plateau
        │
        ├── Save best checkpoint
        ├── Export model (ONNX, TorchScript, etc.)
        ├── Update experiment (status=completed)
        ├── Update model registry (status=completed, metrics)
        └── TrainingBackend.shutdown()
```

## Dataset Loading Flow

1. `DatasetLoader.load_dataset(name, version)` reads quality report from `dataset_quality/reports/`
2. Rejects with `DatasetNotReadyError` if report doesn't exist or `production_ready == False`
3. Returns `DatasetLoadResult` with paths, class names, image counts
4. `TrainingPipeline` resolves dataset YAML path for the training backend

## Augmentation Flow

1. Create `AugmentationConfig` with probabilities for each transform
2. Pass to `AugmentationPipeline.create_pipeline(config)` → returns config dict
3. Config dict is consumed by framework-specific backend at train time

Built-in transforms: Flip, Rotate, Scale, Crop, Brightness, Contrast, Blur, Noise, Mosaic, MixUp

## Hyperparameter Management

Four built-in profiles:

| Profile | LR | Batch | Epochs | Optimizer | Image Size | Description |
|---------|----|-------|--------|-----------|------------|-------------|
| fast | 0.001 | 32 | 10 | adam | 416 | Prototyping |
| balanced | 0.001 | 16 | 100 | adam | 640 | Production |
| accurate | 0.0005 | 8 | 300 | adamw | 768 | High-accuracy |
| tiny | 0.002 | 64 | 50 | sgd | 320 | Edge devices |

`HyperparameterManager.apply_profile(profile, overrides)` applies a profile with optional overrides.

## Trainer Architecture

The `Trainer` depends **only on interfaces** (Dependency Inversion):

```
TrainerInterface
  ├── TrainingBackendInterface ─── initialize(), train_epoch(), validate(), export()
  ├── ExperimentManagerInterface ─── create/get/update experiment
  ├── ModelRegistryInterface ─────── register/update model
  ├── CheckpointManagerInterface ─── save/load/prune checkpoints
  ├── MetricsManagerInterface ────── record/query metrics
  ├── HistoryManagerInterface ────── record history
  ├── EvaluationEngineInterface ──── validate/test/benchmark
  ├── ExportPipelineInterface ────── export to ONNX/TorchScript/OpenVINO
  └── SchedulerInterface ─────────── get_lr() per epoch
```

No concrete implementation is coupled. Add new backends (RT-DETR, DETR, etc.) by implementing `TrainingBackendInterface`.

## Evaluation Flow

1. During training: `TrainingBackend.validate()` with current checkpoint
2. `EvaluationEngine.validate()` persists evaluation result as JSON
3. Result stores mAP@50, mAP@50:95, precision, recall, F1, per-class metrics
4. Full test evaluation and benchmark also supported

## Checkpoint Lifecycle

1. Checkpoints saved every `save_interval` epochs
2. Best checkpoint tracked automatically (highest mAP@50)
3. Latest checkpoint tracked for resume
4. Old checkpoints pruned to `keep_last_n_checkpoints`
5. Checkpoint metadata persisted as JSON

## Experiment Tracking

- `ExperimentManager` creates, reads, updates, lists, deletes experiments
- Each experiment stores: config, timing, best metric, status, notes
- Status lifecycle: `created → running → completed | failed | interrupted`
- Experiments persisted as JSON files

## Model Registry

- `ModelRegistry` registers trained models with full metadata
- Stores: version, framework, architecture, metrics, dataset version, export formats
- Models persisted as JSON files

## Export Workflow

1. After training completes, `TrainingPipeline` exports the model
2. `ExportPipeline` supports ONNX, TorchScript, OpenVINO
3. Actual conversion performed by framework-specific `TrainingBackend.export()`
4. Export metadata persisted as JSON

## Integration with Dataset Quality

- `DatasetLoader` reads quality reports from `backend/dataset_quality/reports/`
- Rejects datasets where `quality_report.overall_score.production_ready == False`
- `TrainingPipeline` stage 1 validates dataset before any training begins

## Integration with Computer Vision Module

- `TrainingResult` stores `model_id` that references `ModelEntry`
- `ModelEntry.checkpoint_path` points to the saved weights
- `ModelEntry.framework` and `architecture` fields enable CV module integration
- Export formats (ONNX, TorchScript) are standard CV deployment formats

## Configuration

All values overridable via environment variables with `TR_` prefix:

```python
class TrainingConfig(BaseSettings):
    model_config = {"env_prefix": "TR_"}
    default_batch_size: int = 16
    default_epochs: int = 100
    default_learning_rate: float = 0.001
    default_optimizer: str = "adam"
    ...
```

## Usage

```python
from backend.training import TrainingService, TrainingConfigData
from backend.training.backends import YOLOTrainingBackend

service = TrainingService(
    training_backend=YOLOTrainingBackend(),
)

config = TrainingConfigData(
    model_name="yolo11n",
    experiment_name="coco_training",
    epochs=100,
    batch_size=16,
)

result = service.train(config)
print(f"Best mAP@50: {result.best_metric}")
print(f"Model ID: {result.model_id}")

# Full pipeline
result = service.run_pipeline(config)
```

## Verification

- **ruff**: `All checks passed!`
- **mypy**: `Success: no issues found`
- **pytest**: `360 passed`
- **Full project**: `1338 passed` (no regressions)
- **Frozen architecture**: Not modified
