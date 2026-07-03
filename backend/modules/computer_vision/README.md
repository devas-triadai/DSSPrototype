# Computer Vision Module

## Purpose

Ingests raw imagery (satellite, drone, reconnaissance) and produces structured object detections for downstream intelligence analysis.

## Architecture

```
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ImageLoader │───▶│ ImageValidator   │───▶│ ImagePreprocessor│
└──────────────┘    └──────────────────┘    └────────┬────────┘
                                                     │
                                                    ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ ResultConv.  │◀───│ InferenceEngine  │◀───│   ModelManager  │
│ (→contracts) │    │                  │    │  (model plugin) │
└──────────────┘    └──────────────────┘    └─────────────────┘
```

## Pipeline

| Step | Component | Responsibility |
|------|-----------|----------------|
| 1 | `ImageLoader` | Decode file path or bytes → RGB NumPy array |
| 2 | `ImageValidator` | Check dimensions, format, integrity |
| 3 | `ImagePreprocessor` | Apply configurable transforms (resize, normalise, …) |
| 4 | `ModelManager` | Retrieve or load a registered model plugin |
| 5 | `InferenceEngine` | Execute model.predict(), measure timing |
| 6 | `ResultConverter` | Map raw output → `DetectionResult` (contracts) |

All steps are coordinated by `ComputerVisionService` which exposes the `VisionModule` interface from `backend.contracts.interfaces`.

## Responsibilities

- Decode and validate image sources
- Apply a configurable preprocessing pipeline
- Acquire a model instance from the model manager
- Execute inference and measure timing
- Convert raw model output to strongly typed `DetectionResult` contracts
- Expose a single `process_image(image: ImageMetadata) -> DetectionResult` entry point

## Inputs

- `ImageMetadata` (contracts) — image identifier, source, timestamp, dimensions
- Image data — filesystem path **or** raw bytes

## Outputs

- `DetectionResult` (contracts) — list of `DetectedObject` instances with types, bounding boxes, and confidence scores

## Dependencies

- `backend.contracts.models.detection`
- `backend.contracts.models.geometry`
- `backend.contracts.interfaces.vision.VisionModule`
- `numpy` — image array representation
- `Pillow` — image decoding and basic transforms

## Extending: Adding a New Vision Model

The system is model-agnostic. To add support for a new model:

1. **Create a class** that implements `VisionModelInterface` (in `backend.modules.computer_vision.interfaces`).
2. **Implement** `load()`, `unload()`, `predict()`, `metadata`, and `status`.
3. **Register** it with the model manager:

```python
from backend.modules.computer_vision.model_manager import register_model

class MyModel(VisionModelInterface):
    ...

register_model("my_model", MyModel)
```

4. **Set** `CV_DEFAULT_MODEL_TYPE=my_model` in your `.env` file.
5. **Optionally** provide a `class_mapping` to `ResultConverter` for automatic `ObjectType` mapping.

No modifications to `service.py`, `inference_engine.py`, or any pipeline component are required.

## Extension Guide

- **New image source** (e.g., stream, URL) → implement `ImageLoaderInterface`
- **New validation rule** → extend `ImageValidator`
- **New preprocessing step** → add a `Transform` callable to the `ImagePreprocessor` constructor
- **New model format** → implement `VisionModelInterface` and register

## Design Philosophy

- **Separation of concerns** — each pipeline step is an isolated, replaceable component
- **Dependency injection** — the service receives its dependencies via constructor; defaults work out of the box
- **Model-agnostic** — no import of YOLO, RT-DETR, or any specific framework in the core pipeline
- **Plugin architecture** — models register themselves; the manager never needs to know concrete types
- **Single responsibility** — `service.py` coordinates but never processes; `inference_engine.py` runs but never converts; `result_converter.py` converts but never infers
