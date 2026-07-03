"""Training backends — model-specific implementations of TrainingBackendInterface.

Every backend (YOLO, Detectron2, MMDetection, RT-DETR, etc.) is a
self-contained plugin that implements TrainingBackendInterface.
The Trainer depends only on the interface, never on concrete backends.
"""

from backend.training.backends.registry import TrainingBackendRegistry
from backend.training.backends.yolo_backend import YOLOTrainingBackend

__all__ = [
    "TrainingBackendRegistry",
    "YOLOTrainingBackend",
]
